"""Implete lattice model with LLL"""
import numpy as np
import matplotlib.pyplot as plt


class Model():
    def __init__(self, M, N, nb=1):
        self.M = M
        self.N = N
        self.nb = nb
        self.kind = ['NN']
        self.phi = 1./3 #0.5
        # deltas[:n] for hoppings up to nth neighbor
        self.deltas = [(1+0j, 0+1j, -1+0j, 0-1j),
                       (1+1j, -1+1j, -1-1j, 1-1j),
                       (2+0j, 0+2j, -2+0j, 0-2j),
                       (2+1j, 1+2j, -1+2j, -2+1j,
                        -2-1j, -1-2j, 1-2j, 2-1j),
                       (2+2j, -2+2j, -2-2j, 2-2j)
                       ]
        self.sites = self.get_sites()
        self.nsite = len(self.sites)
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

    def get_pairs(self, zsite=True):
        """Find hopping pairs for given kind.

        Return complex coordinate if zsite is True
        """
        assert self.kind == ['NN']
        M, N = self.M, self.N
        pairs = []
        for z in self.sites:
            ptmp = []
            # deltas[:n] for hoppings up to nth neighbor
            dzs = (dz for dzz in self.deltas[:1] for dz in dzz)
            for dz in dzs:
                z2 = z + dz
                # coordinate for z2 in PBC
                # z2 = complex(z2.real % M, z2.imag % N)
                ptmp.append(self.pair2int((z2.real % M, z2.imag % N)))
                # if zsite:
                #     ptmp.append(z2)
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

        def J(zj, z):
            x, y = z.real, z.imag
            w = (-1)**(x+y+x*y) * np.exp(-np.pi/2*(1-phi)*abs(z)**2)
            e = np.exp(np.pi/2*(zj*z.conjugate() - zj.conjugate()*z)*phi)
            return w * e
        # sum over NN lattice transiton (Kapit2010). abs(R) = L
        # Rs = (0,)
        Rs = (0, 1*M, 1j*N, -1*M, -1j*N)
        hoppings = []
        # convert to complex coordinate
        pairs = ((self.sites[zk] for zk in zks) for zks in self.pairs)
        for zj, zks in zip(self.sites, pairs):
            tmp = []
            for zk in zks:
                z = zk - zj
                # closest distance is distance
                z = min((z+R for R in Rs), key=lambda z: abs(z))

                hopping = sum(J(zj, z+R) *
                              np.exp(np.pi/2*(zj*R.conjugate()-zj.conjugate()*R)
                                     * phi)
                              for R in Rs)
                hopping = J(zj, z)
                hopping = -1
                tmp.append(hopping)
            hoppings.append(tmp)
        assert len(hoppings) == len(self.sites)
        return hoppings

    def get_basis(self):
        """
        Generate basis sequence with conserved quantum number `nu`.

        Returns
        -------
        basis : array_like
            a list of basis states
        nmems : array_like
            a list containing the number of members for corresponding basis state
        """
        basis = []
        # use `nu`-conserved state generator
    #   for i in xrange(2**nn):
    #       if bin(i).count('1') == nu:
        # first state with `nu` up spins.
        nu = self.nb
        v = 2**nu - 1
        for i in self._samebits_int(v):
                    basis.append(i)
        return basis

    def _samebits_int(self, v):
        """Generate a sequence of integer with same number of 1 bits in binary form.

        The first integer is `v`, and each of these is smaller than ``2**nn``.
        """
        # value 0 treated specially
        if v == 0:
            yield v
            raise StopIteration

        # maximum = 2**nn
        maximum = 2**self.nsite
        while v < maximum:
    #       print('{:0{}b}'.format(v, nn))
            yield v
            t = (v | (v - 1)) + 1
            v = t | ((((t & -t) // (v & -v)) >> 1) - 1)

    def get_hamiltonian(self, mat):
        """Return the hamiltonian of the model"""
        assert self.nb == 1
        # for i, (s, nbrs, hoppings) in enumerate(zip(basis, self.pairs, self.hoppings)):
        #     for nbr, hopping in zip(nbrs, hoppings):
        #         # XXX j
        #        j = basis.index(nbr)
        for i, (js, hoppings) in enumerate(zip(self.pairs, self.hoppings)):
            for j, hopping in zip(js, hoppings):
                mat[i, j] = hopping
                mat[j, i] = hopping.conjugate()
        return mat

if __name__ == '__main__':
    M = 12
    m = Model(M, M, nb=1)
    basis = m.get_basis()
    nst = len(basis)
#   for b in basis[:15]:
#       print format(b, '016b')
#   print len(basis)

    # Construct Hamiltonian matrix
    mat = np.zeros((nst, nst), dtype=complex)
    mat = m.get_hamiltonian(mat)

    # Diagonalize Hamiltonian
    val, vec = np.linalg.eigh(mat)
    plt.plot(val, 'ro')
    plt.show()
