#!/usr/bin/gnuplot

set logscale y
set xlabel "Epoch"
set ylabel "Loss"
set term post eps enhanced color solid size 6,4
set out "loss.eps"

plot "loss.dat" u 0:1 w l title "Train", "" u 0:2 w l title "Test"

!epstopdf loss.eps
!rm loss.eps

