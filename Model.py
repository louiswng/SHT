from turtle import forward
import torch as t
from torch import nn
from Params import args

xavierInit = nn.init.xavier_uniform_
zeroInit = lambda x: nn.init.constant_(x, 0.0)

class SAHT(nn.Module):
    def __init__(self):
        super(SAHT, self).__init__()
        self.LightGCN = LightGCN()#.cuda()
        self.prepareKey = prepareKey()#.cuda()
        self.HypergraphTransormer1 = HypergraphTransormer()#.cuda()
        self.HypergraphTransormer2 = HypergraphTransormer()#.cuda()
        self.label = LabelNetwork()#.cuda()
        
    def forward(self, adj, tpAdj):
        self.uEmbeds0, self.iEmbeds0 = self.LightGCN(adj, tpAdj) # (usr, d)
        self.uKey = self.prepareKey(self.uEmbeds0)
        self.iKey = self.prepareKey(self.iEmbeds0)
        self.ulat, self.uHyper = self.HypergraphTransormer1(self.uEmbeds0, self.uKey)
        self.ilat, self.iHyper = self.HypergraphTransormer2(self.iEmbeds0, self.iKey)
        
    def train(self, uids, iids, edgeids, trnMat):
        pckUlat = self.ulat[uids] # (batch, d)
        pckIlat = self.ilat[iids]
        preds = t.sum(pckUlat * pckIlat, dim=-1) # (batch, batch, d)

        coo = trnMat.tocoo()
        usrs, itms = coo.row[edgeids], coo.col[edgeids]
        self.uKey = t.reshape(t.permute(self.uKey, dims=[1, 0, 2]), [-1, args.latdim])
        self.iKey = t.reshape(t.permute(self.iKey, dims=[1, 0, 2]), [-1, args.latdim])
        usrKey = self.uKey[usrs]
        itmKey = self.iKey[itms]
        scores = self.label(usrKey, itmKey, self.uHyper, self.iHyper)
        _preds = t.sum(self.uEmbeds0[usrs]*self.iEmbeds0[itms], dim=1)

        halfNum = scores.shape[0] // 2
        fstScores = scores[:halfNum]
        scdScores = scores[halfNum:]
        fstPreds = _preds[:halfNum]
        scdPreds = _preds[halfNum:]
        sslLoss = t.sum(t.maximum(t.tensor(0.0), 1.0 - (fstPreds - scdPreds) * args.mult * (fstScores-scdScores)))
        return preds, sslLoss

    def test(self, usr, trnMask):
        pckUlat = self.ulat[usr]
        allPreds = pckUlat @ t.t(self.ilat)
        allPreds = allPreds * (1 - trnMask) - trnMask * 1e8
        _, topLocs = t.topk(allPreds, args.shoot)
        return topLocs


class LightGCN(nn.Module):
    def __init__(self, uEmbeds=None, iEmbeds=None):
        super(LightGCN, self).__init__()

        self.uEmbeds = uEmbeds if uEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.user, args.latdim)))
        self.iEmbeds = iEmbeds if iEmbeds is not None else nn.Parameter(xavierInit(t.empty(args.item, args.latdim)))
        self.gnnLayers = nn.Sequential(*[GCNLayer() for i in range(args.gcn_hops)])
    
    def forward(self, adj, tpAdj):
        ulats = [self.uEmbeds]
        ilats = [self.iEmbeds]
        for gcn in self.gnnLayers:
            temulat = gcn(adj, ilats[-1])
            temilat = gcn(tpAdj, ulats[-1])
            ulats.append(temulat)
            ilats.append(temilat)
        return sum(ulats[1:]) + self.uEmbeds, sum(ilats[1:]) + self.iEmbeds

class GCNLayer(nn.Module):
    def __init__(self):
        super(GCNLayer, self).__init__()
    
    def forward(self, adj, embeds):
        return t.spmm(adj, embeds)

class prepareKey(nn.Module):
    def __init__(self):
        super(prepareKey, self).__init__()
        self.K = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
    
    def forward(self, nodeEmbed):
        key = t.reshape(nodeEmbed @ self.K, [-1, args.att_head, args.latdim//args.att_head])
        key = t.permute(key, dims=[1, 0, 2])
        return key

class prepareValue(nn.Module):
    def __init__(self):
        super(prepareValue, self).__init__()
        self.V = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
    
    def forward(self, nodeEmbed):
        value = t.reshape(nodeEmbed @ self.V, [-1, args.att_head, args.latdim//args.att_head])
        value = t.permute(value, dims=[1, 2, 0])
        return value

class HypergraphTransormer(nn.Module):
    def __init__(self):
        super(HypergraphTransormer, self).__init__()
        self.hypergraphLayers = nn.Sequential(*[HypergraphTransformerLayer() for i in range(args.gnn_layer)])
        self.Hyper = nn.Parameter(xavierInit(t.empty(args.hyperNum, args.latdim)))
        self.V = nn.Parameter(xavierInit(t.empty(args.latdim, args.latdim)))
        self.prepareValue = prepareValue()

    def forward(self, Embed0, Key):
        lats = [Embed0]
        for hypergraph in self.hypergraphLayers:
            Value = self.prepareValue(lats[-1])
            lats = hypergraph(lats, Key, Value, self.Hyper, self.V)
        return sum(lats), self.Hyper

class HypergraphTransformerLayer(nn.Module):
    def __init__(self):
        super(HypergraphTransformerLayer, self).__init__()
        self.linear1 = nn.Linear(args.hyperNum, args.hyperNum, bias=True)
        self.linear2 = nn.Linear(args.hyperNum, args.hyperNum, bias=True)
        self.leakyrelu = nn.LeakyReLU(args.leaky)

    def forward(self, lats, key, value, hyper, V):
        temlat1 = value @ key
        # prepare query
        hyper = t.reshape(hyper, [-1, args.att_head, args.latdim//args.att_head])
        hyper = t.permute(hyper, dims=[1, 2, 0])
        temlat1 = t.reshape(temlat1 @ hyper, [args.latdim, -1])
        temlat2 = self.leakyrelu(self.linear1(temlat1)) + temlat1
        temlat3 = self.leakyrelu(self.linear2(temlat2)) + temlat2

        preNewLat = t.reshape(t.t(temlat3) @ V, [-1, args.att_head, args.latdim//args.att_head])
        preNewLat = t.permute(preNewLat, [1, 0, 2])
        preNewLat = hyper @ preNewLat
        newLat = key @ preNewLat
        newLat = t.reshape(t.permute(newLat, [1, 0, 2]), [-1, args.latdim])
        lats.append(newLat)
        return lats

class Meta(nn.Module):
    def __init__(self):
        super(Meta, self).__init__()
        self.linear1 = nn.Linear(args.latdim, args.latdim * args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, args.latdim, bias=True)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
    def forward(self, hyper):
        hyper_mean = t.mean(hyper, dim=0, keepdim=True)
        hyper = hyper_mean
        W1 = t.reshape(self.linear1(hyper), [args.latdim, args.latdim])
        b1 = self.linear2(hyper)
        def mapping(key):
            ret = self.leakyrelu(key @ W1 + b1)
            return ret
        return mapping

class LabelNetwork(nn.Module):
    def __init__(self):
        super(LabelNetwork, self).__init__()
        self.meta = Meta()
        self.linear1 = nn.Linear(2*args.latdim, args.latdim, bias=True)
        self.linear2 = nn.Linear(args.latdim, 1, bias=True)
        self.leakyrelu = nn.LeakyReLU(args.leaky)
        self.sigmoid = nn.Sigmoid()
    def forward(self, usrKey, itmKey, uHyper, iHyper):
        uMapping = self.meta(uHyper)
        iMapping = self.meta(iHyper)
        ulat = uMapping(usrKey)
        ilat = iMapping(itmKey)
        lat = t.cat((ulat, ilat), dim=1)
        lat = self.leakyrelu(self.linear1(lat)) + ulat + ilat
        ret = t.reshape(self.sigmoid(self.linear2(lat)), [-1])
        return ret


