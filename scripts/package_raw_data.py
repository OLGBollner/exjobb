import sys
from beyblade.parsers import parse_zfs_simulation_dataset

def main():
  data_path = sys.argv[1]
  method = str(sys.argv[2])
  print(f"Collecting data from {data_path}")
  raw_data = parse_zfs_simulation_dataset(data_path, method=method)

  save_path = raw_data.save()

  print(f"Saved raw data in: {save_path}")

if __name__ == "__main__":
  main()
