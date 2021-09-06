if [[ ! -f models/fairseq/de-en/mle/averaged_model.pt ]]; then
  mkdir -p models/fairseq
  echo "Downloading de-en"
  wget -O models/fairseq/de-en.tar https://surfdrive.surf.nl/files/index.php/s/8lnmJPX3iXndRpz/download
  echo "Uncompressing"
  tar -xvf models/fairseq/de-en.tar -C models/fairseq/
  rm models/fairseq/de-en.tar
  echo "Check: models/fairseq/de-en"
else
  echo "Check: models/fairseq/de-en"
fi
