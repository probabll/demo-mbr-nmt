mkdir -p models/fairseq
echo "Downloading en-de"
!wget -O models/fairseq/en-de.tar https://surfdrive.surf.nl/files/index.php/s/TsTcy5oCXFjpmTo/download
echo "Uncompressing"
tar -xvf models/fairseq/en-de.tar -C models/fairseq/
echo "Check: models/fairseq/en-de"
