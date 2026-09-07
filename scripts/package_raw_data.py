from beyblade.parsers import parse_zfs_simulation_dataset, parse_phonon_npz, parse_phonopy_yaml
from beyblade.pipeline import find_default_phonon_file
from pathlib import Path
import argparse

def main():
  parser = argparse.ArgumentParser(description="Package simulation data into a single .npz")

  parser.add_argument("--sim_folder", type=str, nargs="+", help="Path to VASP simulation folder.")
  parser.add_argument("--method", type=str, help="Sets zfs calculation method (all or approx).")
  parser.add_argument("--phonon_file", "-p", type=str, default=None, help="Path to phonon spectrum (.npz or .yaml) to enrich with mode-dependent SI displacements.")

  args = parser.parse_args()
  data_path = args.sim_folder
  method = args.method

  print(f"Collecting data from {data_path}")
  if data_path is not None:
      if isinstance(data_path, (list, tuple)) and len(data_path) > 1:
          datasets = []
          for sf in data_path:
              ds = parse_zfs_simulation_dataset(
                  sim_folder=sf,
                  calc_method=method
              )
              datasets.append(ds)

          raw_data = datasets[0]
          for other_ds in datasets[1:]:
              raw_data = raw_data.combine(other_ds)
      else:
          sf = data_path[0] if isinstance(data_path, (list, tuple)) else data_path
          raw_data = parse_zfs_simulation_dataset(
              sim_folder=sf,
              calc_method=method,
          )
  else:
      raise ValueError("Must provide data_path.")

  spectrum = None
  ph_file = args.phonon_file
  if ph_file is None:
      sf_first = data_path[0] if isinstance(data_path, (list, tuple)) else data_path
      found = find_default_phonon_file(Path(sf_first))
      if found:
          ph_file = str(found)

  if ph_file is not None and Path(ph_file).exists():
      ph_p = Path(ph_file)
      if ph_p.suffix == ".npz":
          spectrum = parse_phonon_npz(ph_p)
      else:
          spectrum = parse_phonopy_yaml(ph_p)
      print(f"Enriching raw data with phonon spectrum from {ph_file}")

  save_path = raw_data.save(spectrum=spectrum)

  print(f"Saved raw data in: {save_path}")

if __name__ == "__main__":
  main()
