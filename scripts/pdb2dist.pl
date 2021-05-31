#!/usr/bin/perl -w
# Badri Adhikari, 3/26/2016
# PDB to CASP RR contacts

use strict;
use warnings;
use Carp;
use Cwd 'abs_path';
use File::Basename;

my $pdb  = shift;
my $atom = shift;
my $sep  = shift;
my $dist = shift;

if (not -f $pdb){
	print STDERR "Usage  : $0 <pdb> <atom> <separation> <dist>\n";
	print STDERR "pdb:$pdb\n";
	exit 1;
}
if (not $atom){
	print STDERR "Usage  : $0 <pdb> <atom> <separation> <dist>\n";
	print STDERR "atom:$atom\n";
	exit 1;
}

# if (not $sep){
# 	print STDERR "Usage  : $0 <pdb> <atom> <separation> <dist>\n";
# 	print STDERR "sep:$sep\n";
# 	exit 1;
# }

our %AA3TO1 = qw(ALA A ASN N CYS C GLN Q HIS H LEU L MET M PRO P THR T TYR Y ARG R ASP D GLU E GLY G ILE I LYS K PHE F SER S TRP W VAL V);
our %AA1TO3 = reverse %AA3TO1;
our @AA3;
our @AA1;

my %rr = ();
my %xyzPDB1 = xyzPDB($pdb, $atom);
my %xyzPDB2 = %xyzPDB1;

foreach my $r1 (sort keys %xyzPDB1){
	foreach my $r2 (sort keys %xyzPDB2){
		next if ($r1 >= $r2);
		next if (($r2 - $r1) < $sep);
		my @row1 = split(/\s+/,$xyzPDB1{$r1});
		my $x1 = $row1[0]; my $y1 = $row1[1]; my $z1 = $row1[2];
		my @row2 = split(/\s+/,$xyzPDB2{$r2});
		my $x2 = $row2[0]; my $y2 = $row2[1]; my $z2 = $row2[2];
		my $d = sqrt(($x1-$x2)**2+($y1-$y2)**2+($z1-$z2)**2);
		$rr{"$r1 $r2 ".(sprintf "%.3f", $d)} = 1;
	}
}

print STDERR "".seqChainWithGaps($pdb)."\n";
foreach (sort keys %rr){
	print "$_\n";
}

################################################################################
# for extracting contacts from native structures
sub seqChainWithGaps{
	my $chain = shift;

	# 1.find end residue number
	my $start; my $end;
	open CHAIN, $chain or die "ERROR! Could not open $chain";
	while(<CHAIN>){
		next if $_ !~ m/^ATOM/;
		next unless (parse_pdb_row($_,"aname") eq "CA");
		confess "ERROR!: ".parse_pdb_row($_,"rname")." residue not defined! \nFile: $chain! \nLine : $_" if (not defined $AA3TO1{parse_pdb_row($_,"rname")});
		$end = parse_pdb_row($_,"rnum");
	}
	close CHAIN;

	# 2.initialize
	my $seq = "";
	for (my $i = 1; $i <= $end; $i++){
		$seq .= "-";
	}
 
	# 3.replace with residues
	open CHAIN, $chain or die "ERROR! Could not open $chain";
	while(<CHAIN>){
		next if $_ !~ m/^ATOM/;
		next unless (parse_pdb_row($_,"aname") eq "CA");
		confess "ERROR!: ".parse_pdb_row($_,"rname")." residue not defined! \nFile: $chain! \nLine : $_" if (not defined $AA3TO1{parse_pdb_row($_,"rname")});
		substr $seq, (parse_pdb_row($_,"rnum") - 1), 1, $AA3TO1{parse_pdb_row($_,"rname")}; 
	}
	close CHAIN;

	confess "$chain has less than 1 residue!" if (length($seq) < 1);
	return $seq;
}

################################################################################
sub parse_pdb_row{
	my $row = shift;
	my $param = shift;
	my $result;
	$result = substr($row,6,5) if ($param eq "anum");
	$result = substr($row,12,4) if ($param eq "aname");
	$result = substr($row,16,1) if ($param eq "altloc");
	$result = substr($row,17,3) if ($param eq "rname");
	$result = substr($row,22,5) if ($param eq "rnum");
	$result = substr($row,21,1) if ($param eq "chain");
	$result = substr($row,30,8) if ($param eq "x");
	$result = substr($row,38,8) if ($param eq "y");
	$result = substr($row,46,8) if ($param eq "z");
	confess "Invalid row[$row] or parameter[$param]" if (not defined $result);
	$result =~ s/\s+//g;
	return $result;
}

################################################################################
sub xyzPDB{
	my $chain = shift;
	my $atom  = shift;
	my %xyzPDB = ();
	open CHAIN, $chain or die "ERROR! Could not open $chain";
	while(<CHAIN>){
		next if $_ !~ m/^ATOM/;
		if ($atom eq "CA"){
			next unless (parse_pdb_row($_,"aname") eq "CA");
		}
		else{
			if(parse_pdb_row($_,"rname") eq "GLY" ){
				next unless (parse_pdb_row($_,"aname") eq "CA");
			}
			else{
				next unless (parse_pdb_row($_,"aname") eq "CB");
			}
		}
		$xyzPDB{"".parse_pdb_row($_,"rnum")} = "".parse_pdb_row($_,"x")." ".parse_pdb_row($_,"y")." ".parse_pdb_row($_,"z");
	}
	close CHAIN;
	confess "ERROR!: xyzPDB is empty\n" if (not scalar keys %xyzPDB);
	return %xyzPDB;
}


################################################################################
sub pdb_atoms{
	my $chain = shift;

	my %atoms = ();
	open CHAIN, $chain or die "ERROR! Could not open $chain";
	while(<CHAIN>){
		next if $_ !~ m/^ATOM/;
		$atoms{parse_pdb_row($_,"rnum")." ".parse_pdb_row($_,"aname")} = 1;
	}
	close CHAIN;
	return %atoms;
}
