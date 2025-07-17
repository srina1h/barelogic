#!/bin/bash

for file in minimally_processed_Hall.csv minimally_processed_Kitchenham.csv minimally_processed_Radjenovic.csv minimally_processed_Wahono.csv
do
    out="${file%.csv}_num.csv"
    awk -F, 'NR==1 {
        printf "%s", $1
        for(i=2; i<=NF; i++) {
            col = $i
            # Remove leading/trailing quotes
            gsub(/^"|"$/, "", col)
            # Capitalize first letter
            first = substr(col,1,1)
            rest = substr(col,2)
            cap = toupper(first) rest
            # Re-quote if original was quoted
            if ($i ~ /^".*"$/) {
                cap = "\"" cap "\""
            }
            printf ",%s", cap
        }
        print ""
        next
    } 1' "$file" > "$out"
    echo "Processed $file -> $out"
done