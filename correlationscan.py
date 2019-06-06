import numpy as np

from modules.eob_inspiral import EOBInspiral

spins = np.linspace(-0.9, 0.9, 18, endpoint=True)
etas = np.logspace(-2, -6, 10, endpoint=True)
chisqs = np.zeros((len(spins), len(etas)))
chisqs = np.insert(chisqs, 0, spins, axis=1)
chisqs = np.insert(chisqs, 0, np.insert(etas, 0, np.nan), axis=0)
for j in range(len(spins)):
    for i in range(len(etas)):
        print('starting inspiral a={} eta={}'.format(spins[j], etas[i]))
        eob = EOBInspiral(spins[j], etas[i], 1.2, correlation_mode='scalewithX')
        chisqs[j + 1][i + 1] = eob.correlate_isco_crossings()

np.savetxt('parameterspacesearch_scale.txt', chisqs, header='spins \\ etas')
