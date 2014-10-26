"""Implete lattice model with LLL"""
import numpy as np
import matplotlib.pyplot as plt
import scipy.linalg as la
# from numba import jit
import bisect


class Model():
    def __init__(self, M, N, nb=1, nbr=1, phi=0.5, bc='pbc', mus={}):
        self.M = M
        self.N = N
        self.nb = nb
        self.phi = phi
        # chemical potential: mus = {site: mu}
        self.mus = mus
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
        self.pairs = self.get_pairs(nbr=nbr)
        self.hoppings = self.get_hoppings(bc=bc)
        self.basis = self.get_basis()

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

    def get_pairs(self, nbr):
        """Find hopping pairs for given kind.

        Return complex coordinate if zsite is True
        """
        M, N = self.M, self.N
        pairs = []
        for zj in self.sites:
            ptmp = []
            # deltas[:n] for hoppings up to nth neighbor
            zs = (z for zs in self.deltas[:nbr] for z in zs)
            for z in zs:
                zk = zj - z
                zk = zk.real % M, zk.imag % N
                ptmp.append((self.pair2int(zk), z))
                # coordinate for z2 in PBC
                # z2 = complex(z2.real % M, z2.imag % N)
                # if zsite:
                #     ptmp.append(z2)
            pairs.append(ptmp)
        assert len(pairs) == len(self.sites)
        return pairs

    def get_hoppings(self, bc):
        """Calculate hopping strength for each pair"""
        M, N = self.M, self.N
        phi = self.phi

        def J(zj, z):
            x, y = z.real, z.imag
            w = (-1)**(x+y+x*y) * np.exp(-np.pi/2*(1-phi)*abs(z)**2)
            # Symmetric guage
            # e = np.exp(np.pi/2*(zj*z.conjugate() - zj.conjugate()*z)*phi)
            # Landau guage, 1xn
            # yj = zj.imag
            # e = np.exp(1j*np.pi*x*(2*yj-y)*phi)
            # Landau guage, nx1
            xj = zj.real
            e = np.exp(1j*np.pi*(x-2*xj)*y*phi)
            return w * e
        # sum over NN lattice transiton (Kapit2010). abs(R) = L
        if bc == 'pbc':
            Rs = (0,)
        # magneto-periodic boundary condition
        elif bc == 'mpbc':
            Rs = (0, 1*M, 1j*N, -1*M, -1j*N)
        else:
            raise NotImplemented

        hoppings = []
        for si, zj in enumerate(self.sites):
            # hoppings.append([J(zj, z) for sk, z in self.pairs[si]])
            # generator will result in wrong answer! consume only once
            es = [np.exp(np.pi/2*(zj*R.conjugate()-zj.conjugate()*R)*phi)
                  for R in Rs]
            hoppings.append([sum(J(zj, z+R)*e for R, e in zip(Rs, es))
                             for sk, z in self.pairs[si]])

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

    def get_index(self, state):
        'Locate the leftmost value exactly equal to state in basis'
        basis = self.basis
        i = bisect.bisect_left(basis, state)
        if i != len(basis) and basis[i] == state:
            return i
        return -1

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
            # print('{:0{}b}'.format(v, nn))
            yield v
            t = (v | (v - 1)) + 1
            v = t | ((((t & -t) // (v & -v)) >> 1) - 1)

    def get_hamiltonian(self, mat):
        """Return the hamiltonian of the model"""
        basis = self.basis
        # get_index = self.get_index
        for ni, n in enumerate(basis):
            for j in xrange(self.nsite):
                for (k, _), t_jk in zip(self.pairs[j], self.hoppings[j]):
                    # if ((n&2**j)>>j, (n&2**k)>>k) == (0, 1):
                    if n & (2**j+2**k) == 2**k:
                        m = n ^ (2**j + 2**k)
                        mi = basis.index(m)
                        # mi, ni = get_index(m), get_index(n)
                        mat[mi, ni] += t_jk
            # chemical potential: mus = {site: mu}
            for site in self.mus:
                # if state & 2**site:
                if n >> site & 1:
                    mat[ni, ni] += self.mus[site]
        return mat

    def get_eigenpair(self, mat):
        "Return eigenvalues and eigenvectors for mat"
        # faster than np.linalg.eigh(mat)
        return la.eigh(mat)


def fig1():
    M, N = 12, 12
    m = Model(M, N, nb=1)
    nst = len(m.basis)
    # Construct Hamiltonian matrix
    mat = np.zeros((nst, nst), dtype=complex)
    fig, ax = plt.subplots(figsize=(8, 4))
    plt.axhline(linestyle='--')
    for nbr, style in zip([1, 2, 4], ['gs', 'ko', 'wo']):
        m = Model(M, M, nb=1, nbr=nbr, phi=1./3)
        mat = m.get_hamiltonian(mat)
        # Diagonalize Hamiltonian
        val, vec = np.linalg.eigh(mat)
        plt.plot(val, style, linestyle='')
        mat[:] = 0
    plt.xlim(-2, 150)
    plt.ylim(-1.1, 1.)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\epsilon_n$')
    # plt.show()
    plt.savefig('data/fig1.pdf')


def fig2():
    M, N = 4, 4
    m = Model(M, N, nb=4, nbr=5, phi=1./2)
    nst = len(m.basis)
    print 'nst: {}'.format(nst)
    # Construct Hamiltonian matrix
    mat = np.zeros((nst, nst), dtype=complex)
    # fig, ax = plt.subplots(figsize=(8, 4))
    mat = m.get_hamiltonian(mat)
    # Diagonalize Hamiltonian
    val, vec = np.linalg.eigh(mat)
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.plot(range(1, 101), val[:100], 'bo', mew=0)
    plt.xlim(-1, 101)
    plt.ylim(-4.05, -2.9)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\epsilon_n$')
    # plt.show()
    plt.savefig('data/fig2.pdf')
    return val, vec


def fig_phis(phis=np.linspace(0.3, 0.5, 20)):
    M, N = 12, 12
    m = Model(M, N, nb=1)
    nst = len(m.basis)
    print 'nst: {}'.format(nst)
    # Construct Hamiltonian matrix
    mat = np.zeros((nst, nst), dtype=complex)
    fig, ax = plt.subplots(figsize=(8, 6))
    for phi in phis:
        m = Model(M, N, phi=phi, nbr=4)
        mat = m.get_hamiltonian(mat)
        # Diagonalize Hamiltonian
        val, vec = np.linalg.eigh(mat)
        plt.plot(phi*np.ones(val.shape), val, '.', ms=1)
        mat[:] = 0
    # plt.xlim(-2, 150)
    # plt.ylim(-1.1, 1.)
    plt.xlabel(r'$\phi$')
    plt.ylabel(r'$\epsilon_n$')
    # plt.show()
    plt.savefig('data/fig_phis.pdf')


def fig_shape():
    iphi = 10
    shapes = [[i]*2 for i in range(5, 26, 5)]
    fig, ax = plt.subplots(figsize=(9, 6))
    for M, N in shapes:
        m = Model(M, N, nbr=4, bc='mpbc', phi=1./iphi)
        nst = len(m.basis)
        print 'nst: {}'.format(nst)
        # Construct Hamiltonian matrix
        mat = np.zeros((nst, nst), dtype=complex)
        # fig, ax = plt.subplots(figsize=(8, 4))
        mat = m.get_hamiltonian(mat)
        # Diagonalize Hamiltonian
        val, vec = np.linalg.eigh(mat)
        plt.plot(val[:100], 'o', mew=0,
                 alpha=0.8, label='shape: {}X{}'.format(M, N))
    plt.title(r'$\phi=1/{}$'.format(iphi))
    plt.legend(loc='best')
    # plt.xlim(-1, 101)
    # plt.ylim(-3.1, -0.9)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\epsilon_n$')
    # plt.show()
    plt.savefig('data/fig_shape.pdf')


def fig_mus():
    M, N = 4, 4
    m = Model(M, N, nb=3, nbr=5, phi=1./2, mus={0:10, 1:10})
    nst = len(m.basis)
    print 'nst: {}'.format(nst)
    # Construct Hamiltonian matrix
    mat = np.zeros((nst, nst), dtype=complex)
    # fig, ax = plt.subplots(figsize=(8, 4))
    mat = m.get_hamiltonian(mat)
    # Diagonalize Hamiltonian
    val, vec = np.linalg.eigh(mat)
    fig, ax = plt.subplots(figsize=(9, 6))
    plt.plot(val[:100], 'bo', mew=0)
    plt.xlim(-1, 101)
    plt.ylim(-3.1, -0.9)
    plt.xlabel(r'$n$')
    plt.ylabel(r'$\epsilon_n$')
    # plt.show()
    plt.savefig('data/fig_mus.pdf')
    return val, vec


if __name__ == '__main__':
    fig_shape()
    # main()
