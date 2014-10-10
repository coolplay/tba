"""Implete lattice model with LLL"""

class Model():
    def __init__(self, M, N):
        self.M = M
        self.N = N
        self.kind = ['NN']
        self.phi = 0.5
        self.sites = self.get_sites()
        self.pairs = self.get_pairs()
        self.hoppings = self.get_hoppings(self.kind)

    def int2pair(self, int):
        """Site index translation"""
        M, N = self.M, self.N
        return (int%M, int//M)

    def pair2int(self, pair):
        """Site index translation"""
        M, N = self.M, self.N
        m, n = (int(i) for i in pair)
        return M * n + m

    def get_sites(self):
        """Set complex coordinates of sites"""
        M, N = self.M, self.N
        sites = []
        for i in range(M*N):
            sites.append(complex(*self.int2pair(i)))
        assert len(sites) == M*N
        return sites

    def get_pairs(self):
        """Find hopping pairs for given kind"""
        assert self.kind == ['NN']
        M, N = self.M, self.N
        pairs = []
        for z in self.sites:
            ptmp = []
            # NN
            for dz in (1, 1j, -1, -1j):
                z2 = z + dz
                #z2 = complex(z2.real % M, z2.imag % N)
                ptmp.append(self.pair2int((z2.real % M, z2.imag % N)))
            pairs.append(ptmp)
        # check for duplicate
        pairs_ = []
        for i, ps in enumerate(pairs):
            tmp = []
            for j in ps:
                if i < j:
                    # will remove j in pairs
                    tmp.append(j)
            pairs_.append(tmp)

        pairs = pairs_
        assert len(pairs) == len(self.sites)
        return pairs

    def get_hoppings(self, kind):
        """Calculate hopping strength for each pair"""
        M, N = self.M, self.N
        phi = self.phi
        import numpy as np
        def J(zj, z):
            x, y = z.real, z.imag
            w = (-1)**(x+y+x*y) * np.exp(-np.pi/2*(1-phi)*abs(z)**2)
            return w*np.exp(np.pi/2*(zj*z.conjugate() - zj.conjugate()*z)*phi)
        # sum over NN lattice transiton. abs(R) = L
        assert self.kind == ['NN']
        hoppings = []
        for zj, zks in zip(self.sites, self.pairs):
            tmp = []
            for zk in zks:
                zk = complex(*self.int2pair(zk))
                z = zk - zj
                hopping = 0
                #XXX r = 0
                for R in (0,):
                # for R in (0, 1*M, 1j*N, -1*M, -1j*N):
                    hopping += J(zj, z+R)*np.exp(np.pi/2*(zj*R.conjugate()-
                        zj.conjugate()*R)*phi)
                tmp.append(hopping)
            hoppings.append(tmp)
        assert len(hoppings) == len(self.sites)
        return hoppings



