#!/bin/bash

total_wait_sec_weekday=0
job_count_weekday=0
total_wait_sec_weekend=0
job_count_weekend=0

while IFS='|' read -r jobid user submit start state rest; do
    if [ "$start" != "None" ]; then
        # Convert timestamps to epoch seconds for calculation
        submit_sec=$(date -d "$submit" +%s)
        start_sec=$(date -d "$start" +%s)
        
        # Calculate wait time in seconds
        wait_sec=$((start_sec - submit_sec))
        
        # Convert to hours:minutes:seconds
        wait_time=$(printf "%d hours, %d minutes, %d seconds" $((wait_sec/3600)) $((wait_sec%3600/60)) $((wait_sec%60)))
        
        # Get day of week (1-7, where 1 is Monday)
        day_of_week=$(date -d "$submit" +%u)
        
        # Separate weekday (1-5) from weekend (6-7)
        if [ $day_of_week -le 5 ]; then
            # Weekday (Monday-Friday)
            total_wait_sec_weekday=$((total_wait_sec_weekday + wait_sec))
            job_count_weekday=$((job_count_weekday + 1))
            echo "Job $jobid: Wait time = $wait_time (Weekday: $(date -d "$submit" +%A))"
        else
            # Weekend (Saturday-Sunday)
            total_wait_sec_weekend=$((total_wait_sec_weekend + wait_sec))
            job_count_weekend=$((job_count_weekend + 1))
            echo "Job $jobid: Wait time = $wait_time (Weekend: $(date -d "$submit" +%A))"
        fi
    fi
done < all_except_single_a100_jobs.txt

# Calculate and display average wait times
echo "-------------------------"
echo "SUMMARY:"

# Weekday average
if [ $job_count_weekday -gt 0 ]; then
    avg_wait_sec_weekday=$((total_wait_sec_weekday / job_count_weekday))
    avg_wait_time_weekday=$(printf "%d hours, %d minutes, %d seconds" $((avg_wait_sec_weekday/3600)) $((avg_wait_sec_weekday%3600/60)) $((avg_wait_sec_weekday%60)))
    echo "Weekday jobs that started: $job_count_weekday"
    echo "Average weekday wait time: $avg_wait_time_weekday"
else
    echo "No weekday jobs found"
fi

# Weekend average
if [ $job_count_weekend -gt 0 ]; then
    avg_wait_sec_weekend=$((total_wait_sec_weekend / job_count_weekend))
    avg_wait_time_weekend=$(printf "%d hours, %d minutes, %d seconds" $((avg_wait_sec_weekend/3600)) $((avg_wait_sec_weekend%3600/60)) $((avg_wait_sec_weekend%60)))
    echo "Weekend jobs that started: $job_count_weekend"
    echo "Average weekend wait time: $avg_wait_time_weekend"
else
    echo "No weekend jobs found"
fi

# Overall average
total_jobs=$((job_count_weekday + job_count_weekend))
if [ $total_jobs -gt 0 ]; then
    total_wait_sec=$((total_wait_sec_weekday + total_wait_sec_weekend))
    avg_wait_sec=$((total_wait_sec / total_jobs))
    avg_wait_time=$(printf "%d hours, %d minutes, %d seconds" $((avg_wait_sec/3600)) $((avg_wait_sec%3600/60)) $((avg_wait_sec%60)))
    echo "-------------------------"
    echo "Total jobs that started: $total_jobs"
    echo "Overall average wait time: $avg_wait_time"
fi
