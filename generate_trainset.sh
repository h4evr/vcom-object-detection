for i in airplane car face motorbike
do
    echo "-$i" >> trainset.txt
    for j in caltech_data/${i}s_train/*.jpg
    do
        echo $j >> trainset.txt
    done
done
