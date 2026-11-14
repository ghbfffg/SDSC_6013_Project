"""Run the whole replication in the order of the paper. Each step writes its own
tables and figures to outputs/stepN_*/. Takes a few minutes (step 2 dominates)."""
import s1_build_dataset, s2_cv_selection, s3_beijing, s4_shanghai, s5_chongqing, s6_summary

if __name__ == "__main__":
    for step in (s1_build_dataset, s2_cv_selection, s3_beijing,
                 s4_shanghai, s5_chongqing, s6_summary):
        print(f"\n===== {step.__name__} =====")
        step.main()
