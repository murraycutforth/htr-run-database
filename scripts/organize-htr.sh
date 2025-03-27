# This script provided by Tony - used in prepare_restart_run.py to keep consistency with the output directory structure
# ==========

# This script takes the *.out output file, console.txt file, and sample0/*
# solution files and places it in $solutiondir.  This is necessary before a
# restart, which always writes to sample0.  In addition, it is a convenient way
# to organize data from multiple runs in a single direcotry.
# Before a restart, be sure to reset "restartDir" in the json file to the desired
# iteration directory .


if [ $# -eq 2 ]; then
   jobid_file=$1
   sampledir=$2   
   if [[ "$sampledir" == *"pp-sample"* ]]; then # for postprocessed hdf files
      solutiondir="postprocess"
   else # for solution files
      solutiondir="solution"
   fi
elif [ $# -eq 3 ]; then
   jobid_file=$1
   sampledir=$2   
   solutiondir=$3
else
   echo "Two or more arguments required: (1) job ID number or file, (2) sample directory, (3) solution directory"
   return
fi
jobid=`basename $jobid_file` # Remove directory name
jobid=${jobid%.*} # Remove .out extension if accidentally included
#jobid=${jobid#'slurm-'} # Remove slurm- part -- only for Yellowstone
jobid=${jobid} # Remove slurm- part -- only for Yellowstone

#echo $jobid

#return

# Check if jobid is a number.  If not, abort.
if ! [[ $jobid =~ ^[0-9]+$ ]] ; then
   echo "**ERROR** jobid=$jobid is invalid.  Aborting."
   return
fi

# Make target directory, if it doesn't exist already
if [ ! -d $solutiondir ]; then
   mkdir -p $solutiondir
fi

# Find common files between the directories
common=$(comm -12 <(ls $sampledir | sort) <(ls $solutiondir | sort))

# Issue warning if $sampledir is not empty
timestamp=$(date +%s%N)
if [ "$common" ]; then
   echo "=========== WARNING ==========="
   echo "Common files exist between $sampledir and $solutiondir:"
   echo ""
   for dir in $common; do
      echo $dir
   done
   echo ""
   echo "==============================="

   # Check with user
   proceed=true
   
   # Overwrite data if requested
   if [ $proceed = true ]; then
      for dir in $common; do
         rm -rv $solutiondir/$dir
         mv -v $sampledir/$dir $solutiondir/.
      done
   fi
fi

# Move non-common fluid_iter solution directories
mv -v $sampledir/fluid_iter* $solutiondir/.

# Move probe files, if any
probe_files=(`find $sampledir -name "probe*.csv" -exec basename {} \;`) # parentheses to make it an array
for f in "${probe_files[@]}"
do
   mv -v "$sampledir/$f" "$solutiondir/$(echo "$f" | sed s/\.csv/-$jobid.csv/)"; # Append file name with jobid
done

# Move console and output files to target directory.
mv -v $sampledir/console.txt $solutiondir/console-$jobid.txt
mv -v $jobid_file $solutiondir/.

# Move grid
mv -v $sampledir/*_grid $solutiondir/.

# Move to autodelete
if [ "$(ls -A $sampledir)" ]; then
   echo "**WARNING** $sampledir is not empty:"
   ls -l $sampledir
else
   rm -rv $sampledir
fi

echo "Done."
