import numpy as np

import em_single

observations = np.array([[1,0,0,0,1,1,0,1,0,1],
                        [1,1,1,1,0,1,1,1,0,1],
                        [1,0,1,1,1,1,1,0,1,1],
                        [1,0,1,0,0,0,1,1,0,0],
                        [0,1,1,1,0,1,1,1,0,1]])

print(em_single.em_run(observations,[0.6,0.5]))