mkdir -p vocabularies svms responses

for i in `seq 12`
do
    echo "Training SVM for conf $i"
    ./bin/trainer confs/$i.txt trainset.txt 500 vocabularies/$i.txt svms/$i.xml.gz responses/$i.txt 2>&1 | tee training_output.txt
done

