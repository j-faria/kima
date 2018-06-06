
template_setup = """

[kima]
obs_after_HARPS_fibers: true / false
GP: true / false
hyperpriors: true / false
trend: true / false

file: filename.txt
units: ms / kms
skip: 0
"""

def need_model_setup():
    print()
    print("[FATAL] Couldn't find the file kima_model_setup.txt")
    print("Probably didn't include a call to save_setup()")
    print("in the RVModel constructor (this is the recommended solution).")
    print("As a workaround, create a file called `kima_model_setup.txt`,")
    print("and add to it (after editting!) the following options:")

    print(template_setup)


