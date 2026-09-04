import sys
from beyblade.parsers import parse_zfs_simulation_dataset

def main():
  data_path = sys.argv[1]
  method = str(sys.argv[2])
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
