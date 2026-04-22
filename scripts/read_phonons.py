import sys
from beyblade.phonon_manager import PhononManager

phonon_mgr = PhononManager(sys.argv[1])
phonon_mgr.save_data()
