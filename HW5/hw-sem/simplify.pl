#!/usr/bin/env perl

use warnings;
use FindBin;
use lib $FindBin::Bin;   # allows finding LambdaTerm module in same directory as this script
use LambdaTerm 'simplify_safe';
use bytes;

while (<>) {
  chomp;
  my($result) = simplify_safe($_);
  if (defined $result) {
    print "Answer: $result\n";
  } else {
    print STDERR "Error: $@";
  }
}
