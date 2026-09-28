from argparse import ArgumentParser as Parser
import re
import numpy as np

parser = Parser("Extracts hyperfine data from vasp OUTCAR")
parser.add_argument("outcar", metavar="outcar")
parser.add_argument("-o", dest="output", default="hyp", help="Name of output file")
args = parser.parse_args()

iso_start_pattern = re.compile(".*\(isotropic\) hyperfine.*")
iso_data_pattern = re.compile("\s*(\d+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+).*")

dip_start_pattern = re.compile(".*Dipolar hyperfine.*")
dip_data_pattern = re.compile("\s*(\d+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+)\s*([-.\d]+)")
pos_data_pattern = re.compile("\s*position of ions in cartesian coordinates.*")

pos_data = []
iso_data = []
dip_data = []
lines = []
with open(args.outcar, "r") as reader:
    lines = reader.readlines()
line_c = 0
while line_c < len(lines):
    line = lines[line_c]

    # capture ion positions
    pos_match = pos_data_pattern.match(line)
    if pos_match is not None:
        line_c += 1
        line = lines[line_c]
        line = line.replace("-"," -")
        vals = line.split()
        while len(vals) == 3:
            line = lines[line_c]
            line = line.replace("-"," -")
            vals = line.split()
            vals = list(map(float,vals))
            pos_data.append(vals)
            line_c += 1

    iso_match = iso_start_pattern.match(line)
    if iso_match is not None:
        ## start gathering isotropic data
        line_c += 4
        line = lines[line_c]
        while len(line.split()) == 6:
            data = line.split()
            ion_ind = int(data[0])
            iso_vals = np.array(list(map(float, data[1:])))
            iso_data.append(iso_vals)
            line_c += 1
            line = lines[line_c]

    dip_match = dip_start_pattern.match(line)
    if dip_match is not None:
        ### start gathering dipole data
        line_c += 4
        line = lines[line_c]
        while len(line.split()) == 7:
            data = line.split()
            ion_ind = int(data[0])
            dip_vals = list(map(float, data[1:]))
            dip_array = np.zeros((3,3))
            dip_array[0,1] = dip_vals[3]
            dip_array[0,2] = dip_vals[4]
            dip_array[1,2] = dip_vals[5]
            dip_array += dip_array.transpose()
            dip_array[0,0] = dip_vals[0]
            dip_array[1,1] = dip_vals[1]
            dip_array[2,2] = dip_vals[2]
            dip_data.append(dip_array)
            line_c += 1
            line = lines[line_c]
    line_c += 1

               
iso_data = np.moveaxis(np.dstack(iso_data), 2, 0)
dip_data = np.moveaxis(np.dstack(dip_data), 2, 0)

true_data = np.zeros((iso_data.shape[0], 4, 3, 3))
for i in range(iso_data.shape[0]):
    iso_tot = iso_data[i,0,-1] + iso_data[i,0,-2] # add total and core contributions to isotropic part
    hyp_tensor = dip_data[i,:, :] + np.eye(3)*iso_tot
    A_ii, v_i = np.linalg.eig(hyp_tensor)
    true_data[i,0,:,:] = np.diag(A_ii)
    true_data[i,1,:,:] = hyp_tensor
    true_data[i,2,:,:] = v_i
    true_data[i,3,:,:] = np.diag(pos_data[i])
np.save(args.output, true_data)
