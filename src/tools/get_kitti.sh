mkdir kitti
cd kitti
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
wget https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip
#wget http://www.cvlibs.net/download.php?file=data_object_image_2.zip
#wget http://www.cvlibs.net/download.php?file=data_object_label_2.zip
#wget http://www.cvlibs.net/download.php?file=data_object_calib.zip
unzip data_object_image_2.zip
unzip data_object_label_2.zip
unzip data_object_calib.zip

