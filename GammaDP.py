import numpy
import decimal

class GammaCalculator:
    
    def __init__(self, weights, nSlots):
        assert nSlots>0, "NSLOTS MUST BE POSITIVE"
        self.nDocs = len(weights)
        self.nSlots = nSlots

        self.nTypes = 0
        self.weightToType = {}
        self.typeToWeight = []
        self.typeToDocs = []
        self.nDocsOfType = []
        self.docToType = []

        self.weights = weights

        for i in range(len(weights)):
            weight = weights[i]
            if not weight in self.weightToType:
                self.typeToWeight.append(decimal.Decimal(weight))
                self.typeToDocs.append([])
                self.nDocsOfType.append(0)
                self.weightToType[weight] = self.nTypes
                self.nTypes += 1

            t = self.weightToType[weight]
            self.docToType.append(t)
            self.nDocsOfType[t] += 1
            self.typeToDocs[t].append(i)

        self.table = {}
        empty_prefix = (0,)*self.nTypes
        self.table[ empty_prefix, () ] = decimal.Decimal(1)
        self.visited = set()
        self.fill_table(empty_prefix, ())

        self.gamma_types = {}
        for (prefix,anchor) in self.table.keys():
            length = sum(prefix)
            for t in range(self.nTypes):
                if prefix[t]<self.nDocsOfType[t]:
                    prob = self.get_prob(prefix, anchor, t)
                    if anchor==():
                        key = "types1", (length,t)
                    else:
                        key = "types2", anchor, (length,t)
                    if not key in self.gamma_types:
                        self.gamma_types[key] = decimal.Decimal(0)
                    self.gamma_types[key] += prob

        self.unitMarginals = numpy.zeros((self.nSlots, self.nDocs), dtype = numpy.longdouble)
        self.pairwiseMarginals = {}
        for (key, prob) in self.gamma_types.items():
            if key[0]=="types1":
                pos, t = key[1]
                normalize = decimal.Decimal(self.nDocsOfType[t])
                for d in self.typeToDocs[t]:
                    self.unitMarginals[pos, d] = numpy.longdouble(prob/normalize)

            if key[0]=="types2":
                pos1, t1 = key[1]
                pos2, t2 = key[2]
                normalize = None
                if t1==t2:
                    normalize = decimal.Decimal(self.nDocsOfType[t1]*(self.nDocsOfType[t2]-1))
                else:
                    normalize = decimal.Decimal(self.nDocsOfType[t1]*self.nDocsOfType[t2])

                newKey = (pos1, pos2)
                if newKey not in self.pairwiseMarginals:
                    self.pairwiseMarginals[newKey] = numpy.zeros((self.nDocs, self.nDocs), dtype = numpy.longdouble)
         
                for d1 in self.typeToDocs[t1]:
                    for d2 in self.typeToDocs[t2]:
                        if d1 != d2:
                            self.pairwiseMarginals[newKey][d1, d2] = numpy.longdouble(prob/normalize)
            
    def decr(self, prefix, t):
        prefix_mut = list(prefix)
        assert prefix_mut[t]>0, "DECR PREFIX OUT OF BOUNDS"
        prefix_mut[t] -= 1
        return tuple(prefix_mut)

    def incr(self, prefix, t):
        prefix_mut = list(prefix)
        assert prefix_mut[t]<self.nDocsOfType[t], "INCR PREFIX OUT OF BOUNDS"
        prefix_mut[t] += 1
        return tuple(prefix_mut)

    def get_prob(self, prefix, anchor, t):
        posterior = [ self.typeToWeight[tt]*(self.nDocsOfType[tt]-prefix[tt]) for tt in range(self.nTypes) ]
        normalize = sum(posterior)
        return self.eval_table(prefix, anchor) * posterior[t] / normalize

    def eval_table(self, prefix, anchor):
        """evaluate an entry in the DP table. here:
              prefix: tuple of type counts
              anchor: specifies (pos, type) where pos<len(prefix)"""

        if (prefix,anchor) in self.table:
            return self.table[prefix,anchor]

        prob = decimal.Decimal(0)
        length = sum(prefix)
        if anchor==() or anchor[0]<length-1:
            for t in range(self.nTypes):
                if prefix[t]>0:
                    prefix0 = self.decr(prefix, t)
                    if anchor==() or prefix0[anchor[1]]>0:
                        prob += self.get_prob(prefix0, anchor, t)
        else:
            t=anchor[1]
            prefix0 = self.decr(prefix, t)
            prob += self.get_prob(prefix0, (), t)
        self.table[prefix,anchor] = prob
        return prob
        
    def fill_table(self, prefix, anchor):
        """add more entries to the DP table extending the current prefix. here:
              prefix: tuple of type counts
              anchor: specifies (pos, type) where pos<len(prefix)"""

        length = sum(prefix)
        if (prefix, anchor) in self.visited:
            return
        self.visited.add( (prefix, anchor) )
        self.eval_table(prefix, anchor)
        if length==self.nSlots-1:
            return

        for t in range(self.nTypes):
            if prefix[t]<self.nDocsOfType[t]:
                prefix1 = self.incr(prefix, t)
                anchor1 = (length, t)
                self.fill_table(prefix1, anchor)
                if anchor==():
                    self.fill_table(prefix1, anchor1)
