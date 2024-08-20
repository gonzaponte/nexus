set -e

folder=$1

for p in 5.0 10.0 15.6; do
  for el in 1 10; do
    for dfh in 0 2 5; do
      for dah in 2.5 5.0 10.0; do
        echo "======== $p | $el | $dfh | $dah ========="
        time python make_psf.py $folder/p_${p}_elgap_${el}_dfh_${dfh}_dah_${dah}_*
      done
    done
  done
done
