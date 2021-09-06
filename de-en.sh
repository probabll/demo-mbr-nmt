echo "Downloading de-en"
mkdir -p models/fairseq
wget https://surfdrive.surf.nl/files/index.php/s/8lnmJPX3iXndRpz/download
echo "Uncompressing"
mv download models/fairseq/de-en.tar
tar -xvf models/fairseq/de-en.tar de-en.tar
