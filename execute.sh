python home_liniar.py > RESULTS.csv
awk -F'\t' -v OFS=',' ' NR == 1461 {print "Id", $0; next}   {print (NR+1460), $0}  ' RESULTS.csv >  RESULTSS.csv
echo -e "Id,SalePrice\n$(cat RESULTSS.csv)" > RESULTSS.csv
