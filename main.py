import csv

from process import extract_info, Person
import glob
import sys
import os

# ------------------------------------------------------------------
# Ovaj fajl ne menjati, da bi automatsko ocenjivanje bilo moguce
if len(sys.argv) > 1:
    VALIDATION_DATASET_PATH = sys.argv[1]
else:
    VALIDATION_DATASET_PATH = os.path.join(os.path.dirname(__file__), 'dataset', 'validation')
# -------------------------------------------------------------------

# izvrsiti citanje teksta sa svih fotografija iz validacionog skupa podataka
processed_image_names = []
extracted_data = []

qr, hb, vb = 0,0,0
for image_path in glob.glob(os.path.join(VALIDATION_DATASET_PATH, "*.bmp")):
    image_directory, image_name = os.path.split(image_path)
    processed_image_names.append(image_name)
    # run predictions
    var = extract_info(image_path)
    extracted_data.append(var)
    if var.company == 'IBM':
        qr += 1
    elif var.company == 'Apple':
        vb += 1
    elif var.company == 'Google':
        hb += 1
print(qr, hb, vb)

# -----------------------------------------------------------------
# Kreiranje fajla sa rezultatima ekstrakcije za svaku sliku

with open('result.csv', 'w') as output_file:
    columns = ['file_name', 'name', 'company', 'date_of_birth', 'job', 'social_security_number']
    writer = csv.DictWriter(output_file,
                            fieldnames=columns,
                            quoting=csv.QUOTE_ALL, delimiter=',', quotechar='"')

    writer.writeheader()
    for file_name, person in zip(processed_image_names, extracted_data):
        writer.writerow({
            'file_name': file_name,
            'name': person.name,
            'company': person.company,
            'date_of_birth': person.date_of_birth.strftime('%Y-%m-%d'),
            'job': person.job,
            'social_security_number': person.ssn
        })
# ------------------------------------------------------------------
