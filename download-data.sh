echo "Downloading"
wget https://surfdrive.surf.nl/files/index.php/s/mVKj1mmX85d7Bs2/download
echo "Uncompressing"
mv download paraphrasing-data-and-models.tgz
tar -xvf paraphrasing-data-and-models.tgz
mv paraphrasing-data-and-models data
rm paraphrasing-data-and-models.tgz
