"""Independent CPU check of NLS score and a periodic analytic solution.

Run: python NLS/audit_score.py (NumPy only). No experiment files are modified.
"""
import numpy as np

N = 1024
L = 20 * np.pi
X = np.arange(N) * L / N
K = 2 * np.pi * np.fft.fftfreq(N, L / N)


def step(p, dt):
    d = np.exp(-1j * K**2 * dt / 2)
    q = np.fft.ifft(np.fft.fft(p) * d)
    return np.fft.ifft(np.fft.fft(q * np.exp(1j * abs(q)**2 * dt)) * d)


def normalize(p):
    return p / np.sqrt(np.mean(abs(p)**2))


def score(p):
    p = normalize(p)
    initial = float(abs(p).max())
    amps = []
    for i in range(5000):
        p = step(p, .01)
        if i % 20 == 0:
            amps.append(float(abs(p).max()))
    mid, late = np.mean(amps[150:200]), np.mean(amps[200:])
    peak = max(amps)
    value = peak * (1 + max(0, (late-mid)/mid)) * (1.5 if amps[-1] > 1.5 else 1)
    return initial, peak, float(value)


def published_ic():
    t, kappa, phase = -4, .72, 3*np.pi/4
    nu = kappa/np.sqrt(2)
    perg = np.exp(1j*t)*(1-4*(1+2j*t)/(1+4*X**2+4*t*t))
    akm = (nu-nu*np.cos(kappa*X+phase))/(1-nu*np.cos(kappa*X+phase))
    return perg+akm-1


def exact_breather(t):
    # Coefficient-one focusing NLS, a=1, phi=pi/4, k=1, growth=1.
    c = np.cos(np.pi/4)*np.cos(X)
    return np.exp(1j*t)*(np.cosh(t-1j*np.pi/2)-c)/(np.cosh(t)-c)


def convergence():
    errors = []
    for dt in (.04, .02, .01, .005):
        p = exact_breather(-2)
        for _ in range(round(4/dt)):
            p = step(p, dt)
        error = float(np.sqrt(np.mean(abs(p-exact_breather(2))**2)))
        errors.append(error)
        print('breather dt, L2 error', dt, error)
    assert all(3.8 < a/b < 4.2 for a,b in zip(errors,errors[1:]))


if __name__ == '__main__':
    convergence()
    original = score(published_ic())
    print('published IC: initial peak, peak, score', original)
    assert abs(original[2]-7.7107) < .002
    target = normalize(np.exp(-((X-L/2)/.2)**2/2)).astype(complex)
    p = target.copy()
    for _ in range(101):
        p = step(p, -.01)
    recovered = p.copy()
    for _ in range(101):
        recovered = step(recovered, .01)
    assert np.max(abs(recovered-target)) < 1e-8
    control = score(p)
    print('reverse control: initial peak, peak, score', control)
    assert control[0] > 10 and control[2] > 19
