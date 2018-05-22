from ctypes import *
import random
import time

iviumdll = windll.LoadLibrary("./IVIUM_remdriver.dll")

# Open the driver
iviumdll.IV_open()

# define argument types
iviumdll.IV_connect.argtypes = [POINTER(c_int)]
iviumdll.IV_savedata.argtypes = [c_char_p]
iviumdll.IV_setmethodparameter.argtypes = [c_char_p, c_char_p]
iviumdll.IV_startmethod.argtypes = [c_char_p]
# define return types
iviumdll.IV_connect.restype = c_int32
iviumdll.IV_connect.IV_getdevicestatus = c_int32


if iviumdll.IV_getdevicestatus() == -1:
    print("No IviumSoft")
if iviumdll.IV_getdevicestatus() == 0:
    iviumdll.IV_connect(byref(c_int(1)))

# Some parameters are already set in iviumsoft
num_samples = 10
for i in range(num_samples):
    label = random.randint(0, 2)
    potential = 0.01 * label
    iviumdll.IV_setmethodparameter(b"Levels.E[1]", str(potential).encode('utf-8'))
    # print(iviumdll.IV_getdevicestatus())
    iviumdll.IV_startmethod(b"")
    # busy waiting
    while True:
        if iviumdll.IV_getdevicestatus() == 1:
            iviumdll.IV_savedata(("testiviumdll_training00"+str(label)+'_'+str(i)).encode('utf-8'))
            break
        time.sleep(0.01)