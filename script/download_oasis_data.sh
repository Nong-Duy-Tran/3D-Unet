#!/bin/bash

# Base URL for downloading OASIS-1 data
BASE_URL="oasis_cross-sectional_disc"

# Loop through disc numbers 2 to 12
for i in {2..12}
do
    # Construct the download URL
    URL="${BASE_URL}${i}.tar.gz"

    # Download the file
    echo "Extract OASIS-1 disc $i from $URL..."
    tar -xzf $URL -C data/OASIS/

done

echo "Extract complete."