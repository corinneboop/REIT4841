### STEP 1a - Unzip the nested AEMO rooftop PV files (outer weekly zip -> inner half-hourly zips -> CSV).
### Input: raw_pv_data_zip/   Output: raw_pv_data/   (already-extracted CSVs are skipped)

import os
import zipfile
from io import BytesIO

UNZIP_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_FOLDER = os.path.join(UNZIP_DIR, 'raw_pv_data_zip')
OUTPUT_FOLDER = os.path.join(UNZIP_DIR, 'raw_pv_data')

os.makedirs(OUTPUT_FOLDER, exist_ok=True)
n_extracted, n_skipped = 0, 0

for outer_filename in os.listdir(BASE_FOLDER):
    if outer_filename.endswith('.zip'):
        outer_path = os.path.join(BASE_FOLDER, outer_filename)

        with zipfile.ZipFile(outer_path, 'r') as outer_zip:
            for inner_name in outer_zip.namelist():
                if inner_name.endswith('.zip'):
                    inner_bytes = outer_zip.read(inner_name)

                    with zipfile.ZipFile(BytesIO(inner_bytes)) as inner_zip:
                        for file in inner_zip.namelist():
                            if file.endswith('.csv'):
                                output_path = os.path.join(OUTPUT_FOLDER, os.path.basename(file))

                                if os.path.exists(output_path):
                                    n_skipped += 1
                                    continue

                                with inner_zip.open(file) as source, open(output_path, 'wb') as target:
                                    target.write(source.read())
                                n_extracted += 1
print(f"Extracted {n_extracted} new CSVs, skipped {n_skipped} already present.")
