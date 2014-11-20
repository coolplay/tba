from fabric.api import local, run, cd, put

src = 'square.py'
rdir = '~/run/physics/tba'


def execute():
    local('python square.py')


def deploy():
    # make sure cmds in run always return 0
    run('[ ! -d {0} ] && mkdir {0}; true'.format(rdir))
    with cd(rdir):
        put(src, rdir)
        data = run('python square.py')
        # save to current folder stdout from remote host
        open('out.dat', 'w').write(data)


def binomial(m=96, n=2):
    # args from cmdline are strings
    m, n = int(m), int(n)
    import sympy
    print '\nnchoosek({}, {}) = {}'.format(m, n, sympy.binomial(m, n))
