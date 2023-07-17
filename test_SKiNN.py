import SKiNN
import torch
import pickle
import numpy as np
assert(torch.cuda.is_available()==True) #must be True to function
from SKiNN.generator import Generator
generator=Generator()
example_input = [9.44922512e-01, 8.26468232e-01, 1.00161407e+00, 3.10945081e+00, 7.90308638e-01, 1.00000000e-04, 4.60606795e-01, 2.67345695e-01, 8.93001866e+01]
vrms_map=np.array(generator.generate_map(example_input))

with open("test_vrms_map", "rb") as fp:   # Unpickling
    b = pickle.load(fp)
b=np.array(b)

np.testing.assert_allclose(vrms_map[:50,:50],b[:50,:50],rtol=1e-3)

print('Tests passed.')
