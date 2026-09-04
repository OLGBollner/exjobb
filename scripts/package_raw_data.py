from beyblade.parsers import parse_zfs_simulation_dataset
import argparse

def main():
  parser = argparse.ArgumentParser(description="Package simulation data into a single .npz")

  parser.add_argument("--sim_folder", type=str, nargs="+", help="Path to VASP simulation folder.")
  parser.add_argument("--method", type=str, help="Sets zfs calculation method (all or approx).")

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

  save_path = raw_data.save()

  print(f"Saved raw data in: {save_path}")

if __name__ == "__main__":
  main()
