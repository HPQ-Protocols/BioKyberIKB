source pqc_env/bin/activate
python BioKyberIKB_Experiment.py
pip install liboqs-python cryptography numpy pandas matplotlib
python experiments/BioKyberIKB_Experiment.py --runs 1000 --warmup 100 --correctness-runs 100
python BioKyberIKB_Experiment.py --runs 1000 --warmup 100 --correctness-runs 100
python BioKyberIKB_Experiment.py --runs 5000 --warmup 200 --fe-samples-csv fingerprint_fe_samples.csv --require-fe
python -m pip install pyfing tensorflow keras opencv-python pillow numpy pandas
python FingerprintFE_Samples.py --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip --dry-run
python FingerprintFE_Samples.py --self-test
python FingerprintFE_Samples.py --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip --output fingerprint_fe_samples.csv
python BioKyberIKB_Experiment.py --runs 5000 --warmup 200 --fe-samples-csv fingerprint_fe_samples.csv --require-fe
source pqc_env/bin/activate
python FingerprintFE_Samples.py --self-test
python FingerprintFE_Samples.py --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip --impostor-mode all-images --output fingerprint_fe_samples_v2.csv
python FingerprintFE_Samples.py --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip --impostor-mode all --output fingerprint_fe_samples_v2.csv
python BioKyberIKB_Experiment.py --runs 5000 --warmup 200 --correctness-runs 100 --fe-samples-csv fingerprint_fe_samples_v2.csv --require-fe
python FingerprintFE_Samples.py --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip --impostor-mode all-images --output fingerprint_fe_samples_v3.csv
python BioKyberIKB_Experiment.py --runs 5000 --warmup 200 --correctness-runs 100 --fe-samples-csv fingerprint_fe_samples_v3.csv --require-fe
python FingerprintFE_Samples.py --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip --output fingerprint_fe_samples_final.csv
source pqc_env/bin/activate
python FingerprintFE_Samples.py --dataset-zips DB1_B.zip DB2_B.zip DB3_B.zip DB4_B.zip --impostor-mode all-images --output fingerprint_fe_samples_final.csv
ls -lh DB?_A.zip DB?_B.zip
zipinfo -1 HDA_COLFISPOOF*.zip | head -200 > colfispoof_structure.txt
zipinfo -1 hda_colfispoof.zip | head -200 > colfispoof_structure.txt
7z l MOLF > molf_structure.txt
sudo apt install p7zip-full
7z l MOLF > molf_structure.txt
7z l MOLF.7z > molf_structure.txt
cls
clear
python MOLF_MatcherGate_Samples.py --molf-root MOLF --sensors DB1_Lumidgm --development-subjects 20 --impostors-per-identity 20 --cache-dir molf_nbis_cache_db1 --output molf_db1_gate_samples.csv
