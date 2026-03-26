import argparse
import pandas as pd
import matplotlib.pyplot as plt

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="*_features.parquet")
    ap.add_argument("--outdir", default="data/raw_telemetry", help="output directory for PNGs")
    args = ap.parse_args()

    df = pd.read_parquet(args.infile).copy()
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    base = args.infile.split("/")[-1].replace("_features.parquet","")

    # 1) Power over time
    plt.figure()
    plt.plot(df["ts_utc"], df["power_w"])
    plt.xlabel("Time (UTC)")
    plt.ylabel("Power (W)")
    plt.title("GPU Power over Time")
    out1 = f"{args.outdir}/{base}_power.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=150)
    plt.close()

    # 2) SM clock over time
    plt.figure()
    plt.plot(df["ts_utc"], df["sm_clock_mhz"])
    plt.xlabel("Time (UTC)")
    plt.ylabel("SM Clock (MHz)")
    plt.title("SM Clock over Time")
    out2 = f"{args.outdir}/{base}_sm_clock.png"
    plt.tight_layout()
    plt.savefig(out2, dpi=150)
    plt.close()

    # 3) Scatter: power vs mem_used_frac (colored by label via separate plots to avoid seaborn)
    plt.figure()
    for lab in sorted(df["label"].unique()):
        sub = df[df["label"] == lab]
        plt.scatter(sub["mem_used_frac"], sub["power_w"], label=lab, s=20, alpha=0.8)
    plt.xlabel("Memory Used Fraction")
    plt.ylabel("Power (W)")
    plt.title("Power vs Memory Footprint by Label")
    plt.legend()
    out3 = f"{args.outdir}/{base}_scatter.png"
    plt.tight_layout()
    plt.savefig(out3, dpi=150)
    plt.close()

    print("Wrote:")
    print(out1)
    print(out2)
    print(out3)

if __name__ == "__main__":
    main()

