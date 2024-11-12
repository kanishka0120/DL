class gates():
    def _init_(self, not_weight = np.array([1]), and_weight = np.array([-1,1])
                , or_weight = np.array([1,-4]), or_biase = 1, and_biase = -7, not_biase = .1,
                learning_rate = .1) -> None:
        self.not_weight = not_weight
        self.and_weight = and_weight
        self.or_weight = or_weight
        self.or_biase = or_biase
        self.and_biase = and_biase
        self.not_biase = not_biase
        self.learning_rate = learning_rate

    def activation(self,x):
        return 1 if x>=0 else 0
    
    def perseptron(self,x,w,b):
        res = np.dot(x,w) + b
        return self.activation(res)
    
    def NOT_function(self,x):
        x = np.array(x)
        return self.perseptron(x,self.not_weight,self.not_biase)
    
    def AND_function(self,x):
        x = np.array(x)
        return self.perseptron(x,self.and_weight,self.and_biase)
    
    def OR_function(self,x):
        x = np.array(x)
        return self.perseptron(x,self.or_weight,self.or_biase)
    
    def XOR_function(self,x):
        x = np.array(x)
        
        y0 = self.NOT_function(x[0])
        y1 = self.NOT_function(x[1])
        z1 = self.AND_function([y0,x[1]])
        z2 = self.AND_function([y1,x[0]])
        
        return self.OR_function([z1,z2])
    
    def NXOR_function(self,x):
        return self.NOT_function(self.XOR_function(x))
    
    def update_weight(self,inp,out,tar,weight):
        res = weight + self.learning_rate*(tar - out)*inp
        return res
    
    def update_biase(self,b,out,tar,):
        res = b + 0.3*(tar - out)
        return res

    def update_not(self):
        inp = np.array([1,0])
        tar = np.array([0,1])
        
        i = 0
        while i <len(inp):
            out = self.NOT_function(inp[i])
            if(tar[i] != out):
                weight = self.not_weight
                self.not_weight = self.update_weight(inp[i],out,tar[i],self.not_weight)
                if (weight == self.not_weight).all():
                    self.not_biase = self.update_biase(self.not_biase,out,tar[i])
                print(self.not_weight,self.not_biase)
                i = 0
            else:
                i += 1
                
    def update_and(self):
        inp = np.array([(0,0),(1,0),(0,1),(1,1)])
        tar = np.array([0,0,0,1])
        
        i = 0
        while i <len(inp):
            out = self.AND_function(inp[i])
            if(tar[i] != out):
                weight = self.and_weight
                self.and_weight = self.update_weight(inp[i],out,tar[i],self.and_weight)
                if (weight == self.and_weight).all():
                    self.and_biase = self.update_biase(self.and_biase,out,tar[i])
                print(self.and_weight,self.and_biase)
                i = 0
            else:
                i += 1
                
    def update_or(self):
        inp = np.array([(0,0),(1,0),(0,1),(1,1)])
        tar = np.array([0,1,1,1])
        
        i = 0
        while i <len(inp):
            out = self.OR_function(inp[i])
            if(tar[i] != out):
                weight = self.or_weight
                self.or_weight = self.update_weight(inp[i],out,tar[i],self.or_weight)
                if (weight == self.or_weight).all():
                    self.or_biase = self.update_biase(self.or_biase,out,tar[i])
                print(self.or_weight,self.or_biase)
                i = 0
            else:
                i += 1
  