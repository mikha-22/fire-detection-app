begin
    head -n 1 peatland_1km_buffer_clusters_2019.csv
    tail -n +2 -q *.csv
end >../all_peatland_clusters_prioritized.csv
