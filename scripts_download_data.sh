#!/usr/bin/env bash
# Downloads the public datasets used by orbit-lab into data-raw/.
# All sources are public; URLs verified 2026-11-08.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p data-raw

# CelesTrak current TLEs (3-line format)
curl -sSL 'https://celestrak.org/NORAD/elements/gp.php?GROUP=stations&FORMAT=tle' -o data-raw/tle-stations.txt
curl -sSL 'https://celestrak.org/NORAD/elements/gp.php?GROUP=starlink&FORMAT=tle' -o data-raw/tle-starlink.txt
curl -sSL 'https://celestrak.org/NORAD/elements/gp.php?GROUP=visual&FORMAT=tle' -o data-raw/tle-visual.txt

echo "Downloaded:"
wc -l data-raw/*.txt

# Natural Earth 110m land polygons (public domain) for map backgrounds
curl -sSL 'https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_110m_land.geojson' -o data-raw/ne110m-land.geojson
