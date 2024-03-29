from Params import args
import os
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
import torch as t
import Utils.TimeLogger as logger
from Utils.TimeLogger import log
from Model import SHT
from DataHandler import DataHandler, negSamp
import numpy as np
import pickle
# import nni
# from nni.utils import merge_parameter
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter(log_dir='runs')

class Recommender:
    def __init__(self, handler):
        self.handler = handler
        print('USER', args.user, 'ITEM', args.item)
        print('NUM OF INTERACTIONS', len(self.handler.trnMat.data))
        # print('NUM OF USER-USER EDGE', args.uuEdgeNum)
		
        self.metrics = dict()
        mets = ['Loss', 'preLoss', 'Recall', 'NDCG']
        for met in mets:
            self.metrics['Train' + met] = list()
            self.metrics['Test' + met] = list()

    def makePrint(self, name, ep, reses, save):
        ret = 'Epoch %d/%d, %s: ' % (ep, args.epoch, name)
        for metric in reses:
            val = reses[metric]
            ret += '%s = %.4f, ' % (metric, val)
            tem = name + metric
            if save and tem in self.metrics:
                self.metrics[tem].append(val)
        ret = ret[:-2] + '  '
        return ret

    def run(self):
        self.prepareModel()
        log('Model Prepared')
        if args.load_model != None:
            self.loadModel()
            stloc = len(self.metrics['TrainLoss']) * args.tstEpoch - (args.tstEpoch - 1)
        else:
            stloc = 0
            log('Model Initialized')
        
        best_acc = 0.0
        for ep in range(stloc, args.epoch):
            tstFlag = (ep % args.tstEpoch == 0)
            reses = self.trainEpoch()
            writer.add_scalar('Loss/train', reses['Loss'], ep)
            log(self.makePrint('Train', ep, reses, tstFlag))
            if tstFlag:
                reses = self.testEpoch()
                if reses['Recall'] > best_acc:
                    best_acc = reses['Recall']
                    best_acc2 = reses['NDCG']
                    es = 0
                else:
                    es += 1
                    if es >= args.patience:
                        print("Early stopping with best Recall and NDCG is", best_acc, best_acc2)
                        break
                writer.add_scalar('Recall/test', reses['Recall'], ep)
                writer.add_scalar('Ndcg/test', reses['NDCG'], ep)
                # nni.report_intermediate_result(reses['Recall'])
                log(self.makePrint('Test', ep, reses, tstFlag))
                self.saveHistory()
            self.sche.step()
            print()
        # reses = self.testEpoch()
        # nni.report_final_result(best_acc)
        # log(self.makePrint('Test', args.epoch, reses, True))
        self.saveHistory()

    def prepareModel(self):
        self.model = SHT().cuda()
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sche = t.optim.lr_scheduler.ExponentialLR(self.opt, gamma=args.decay)

    def sampleTrainBatch(self, batIds, labelMat):
        temLabel = labelMat[batIds].toarray()
        batch = len(batIds)
        temlen = batch * 2 * args.sampNum
        uLocs = [None] * temlen
        iLocs = [None] * temlen
        cur = 0
        for i in range(batch):
            posset = np.reshape(np.argwhere(temLabel[i]!=0), [-1])
            sampNum = min(args.sampNum, len(posset))
            if sampNum == 0:
                poslocs = [np.random.choice(args.item)]
                neglocs = [poslocs[0]]
            else:
                poslocs = np.random.choice(posset, sampNum)
                neglocs = negSamp(temLabel[i], sampNum, args.item)
            for j in range(sampNum):
                posloc = poslocs[j]
                negloc = neglocs[j]
                uLocs[cur] = uLocs[cur+temlen//2] = batIds[i]
                iLocs[cur] = posloc
                iLocs[cur+temlen//2] = negloc
                cur += 1
        uLocs = uLocs[:cur] + uLocs[temlen//2: temlen//2 + cur]
        iLocs = iLocs[:cur] + iLocs[temlen//2: temlen//2 + cur]
        
        edgeSampNum = int(args.edgeSampRate * args.edgeNum)
        if edgeSampNum % 2 == 1:
            edgeSampNum += 1
        edgeids = np.random.choice(args.edgeNum, edgeSampNum)
        return uLocs, iLocs, edgeids

    def trainEpoch(self):
        num = args.user
        sfIds = np.random.permutation(num)[:args.trnNum]
        epLoss, epPreLoss, epsslLoss = [0] * 3
        num = len(sfIds)
        steps = int(np.ceil(num / args.batch))
        self.adj = self.handler.torchAdj
        self.tpAdj = self.handler.torchTpAdj
        self.model.train()
        for i in range(steps):
            st = i * args.batch
            ed = min((i+1) * args.batch, num)
            batIds = sfIds[st: ed]

            uLocs, iLocs, edgeids = self.sampleTrainBatch(batIds, self.handler.trnMat)
            preLoss, sslLoss = self.model.calcLosses(self.adj, self.tpAdj, uLocs, iLocs, edgeids, self.handler.trnMat)

            regLoss = 0
            for W in self.model.parameters():
                regLoss += W.norm(2).square()   
            regLoss *= args.reg
            sslLoss = args.ssl_reg * sslLoss
            loss = preLoss + regLoss + sslLoss            
            epLoss += loss.item()
            epPreLoss += preLoss.item()
            epsslLoss += sslLoss.item()

            self.opt.zero_grad()
            loss.backward()
            self.opt.step()

            log('Step %d/%d: loss = %.2f, regLoss = %.2f, sslLoss = %.2f         ' % (i, steps, loss, regLoss, sslLoss), save=False, oneline=True)
        
        ret = dict()
        ret['Loss'] = epLoss / steps
        ret['preLoss'] = epPreLoss / steps
        ret['sslLoss'] = epsslLoss / steps
        return ret

    def testEpoch(self):
        tstLoader = self.handler.tstLoader
        epLoss, epRecall, epNdcg = [0] * 3
        i = 0
        num = tstLoader.dataset.__len__()
        steps = num // args.tstBat
        for usr, trnMask in tstLoader:
            i += 1
            usr = usr.long().cuda()
            trnMask = trnMask.cuda()
            
            allPreds = self.model.predAll(self.handler.torchAdj, self.handler.torchTpAdj, usr)
            allPreds = allPreds * (1 - trnMask) - trnMask * 1e8
            
            _, topLocs = t.topk(allPreds, args.topk)
            recall, ndcg = self.calcRes(topLocs.cpu().numpy(), self.handler.tstLoader.dataset.tstLocs, usr)
            epRecall += recall
            epNdcg += ndcg
            log('Steps %d/%d: recall = %.2f, ndcg = %.2f          ' % (i, steps, recall, ndcg), save=False, oneline=True)    
        ret = dict()
        ret['Recall'] = epRecall / num
        ret['NDCG'] = epNdcg / num
        return ret

    def calcRes(self, topLocs, tstLocs, batIds):
        assert topLocs.shape[0] == len(batIds)
        allRecall = allNdcg = 0
        recallBig = 0
        ndcgBig =0
        for i in range(len(batIds)):
            temTopLocs = list(topLocs[i])
            temTstLocs = tstLocs[batIds[i]]
            tstNum = len(temTstLocs)
            maxDcg = np.sum([np.reciprocal(np.log2(loc + 2)) for loc in range(min(tstNum, args.topk))])
            recall = dcg = 0
            for val in temTstLocs:
                if val in temTopLocs:
                    recall += 1
                    dcg += np.reciprocal(np.log2(temTopLocs.index(val) + 2))
            recall = recall / tstNum
            ndcg = dcg / maxDcg
            allRecall += recall
            allNdcg += ndcg
        return allRecall, allNdcg

    def saveHistory(self):
        if args.epoch == 0:
            return
        with open('History/' + args.save_path + '.his', 'wb') as fs:
            pickle.dump(self.metrics, fs)

        content = {
			'SHT': self.model
		}
        t.save(content, 'Models/' + args.save_path + '.mod')
        log('Model Saved: %s' % args.save_path)

    def loadModel(self):
        ckp = t.load('Models/' + args.load_model + '.mod')
        self.model = ckp['SHT']
        
        with open('History/' + args.load_model + '.his', 'rb') as fs:
            self.metrics = pickle.load(fs)
        self.opt = t.optim.Adam(self.model.parameters(), lr=args.lr)
        self.sche = t.optim.lr_scheduler.ExponentialLR(self.opt, gamma=args.decay)
        log('Model Loaded')	

if __name__ == '__main__':
    logger.saveDefault = True
    device = "cuda" if t.cuda.is_available() else "cpu"
    print(f"Using {device} device")

    # get parameters form tuner
    # tuner_params = nni.get_next_parameter()
    # params = vars(merge_parameter(args, tuner_params))
    # print(params)
    
    log('Start')
    handler = DataHandler()
    handler.LoadData()
    log('Load Data')
    
    recom = Recommender(handler)
    recom.run()