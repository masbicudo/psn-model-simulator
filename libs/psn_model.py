import datetime as dt
import os
import numpy as np
import pandas as pd

from typing import Literal, Set

import libs.data_helpers as dat

from libs.utils import any_opt, Memoize

def get_shot_series_from_events(
            key,
            events,
            log: Set[Literal["none", "all"]]="none",
        ):
    min_value = min(events)
    max_value = max(events)

    if any_opt(log, ["all"]): print(f"{key}")
    if any_opt(log, ["all"]): print(f"    min_value = {min_value}")
    if any_opt(log, ["all"]): print(f"    max_value = {max_value}")

    # one bin for each day
    bins = [min_value + dt.timedelta(days=x)
            for x in range((max_value - min_value).days + 2)]
    hist, bins = np.histogram(events, bins=bins)
    return hist, bins

@Memoize
def get_shots(
        df,
        shot_id_column,
        event_date_column,
        log: Set[Literal["none", "all"]]="none",
        ):
    from tqdm import tqdm
    progress = tqdm if any_opt(log, ["some", "all"]) else lambda x: x
    shots = {}
    for k, t in progress(df.groupby(shot_id_column)):
        hist, bins = get_shot_series_from_events(k, t[event_date_column], log)
        shots[k] = pd.Series(np.array(hist), np.array(bins[:-1]))
    return shots

@Memoize
def filter_shots_series_by_date(shots, filter, cut_on_border=True):
    result = {}
    count = 0
    #from tqdm import tqdm
    tqdm = lambda x:x
    for k in tqdm(shots):
        s:pd.Series = shots[k]
        i:pd.DatetimeIndex = s.index
        count += filter(i[0])
        f = filter(i)
        if cut_on_border:
            s = s[f]
        if f.any() and len(s) > 0:
            result[k] = s
    return result, count

def filter_series_by_date_index(series:pd.Series, filter):
    if isinstance(series.index, pd.DatetimeIndex):
        f = filter(series.index.date)
    else:
        f = filter(series.index)
    series = series[f]
    return series

transform_ones = lambda v: v*0 + 1

@Memoize
def transform_shots_series(shots, transformation):
    result = {}
    for k in shots:
        s:pd.Series = shots[k]
        s2 = transformation(s)
        result[k] = s2
    return result

from bisect import bisect_left
def get_aggregated_flow(
        shots: list,
        log: Set[Literal["none", "some", "all"]]="none",
    ):
    if len(shots) == 0:
        raise Exception("shots argument is empty")
    min_value_total = min(shots[k].index[0] for k in shots)
    max_value_total = max(shots[k].index[-1] for k in shots)
    if any_opt(log, ["some", "all"]): print(f"min_value_total = {min_value_total}")
    if any_opt(log, ["some", "all"]): print(f"max_value_total = {max_value_total}")
    # Note: we need the extra +1 because sometimes there is an error in
    # the calculation of (max_value_total - min_value_total), when these values
    # have hours set to different values that should be rounded to an extra day
    bins_total = [min_value_total.date() + dt.timedelta(days=x)
            for x in range((max_value_total - min_value_total).days + 2 +1)]
    total_hist = np.zeros((max_value_total - min_value_total).days + 1 +1)
    if any_opt(log, ["some", "all"]): print(f"len(bins_total) = {len(bins_total)}")
    if any_opt(log, ["some", "all"]): print(f"len(total_hist) = {len(total_hist)}")
    if any_opt(log, ["some", "all"]): print(f"len(shots) = {len(shots)}")

    for k in shots:
        s = shots[k]
        min_value = s.index[0]
        pos = bisect_left(bins_total, min_value.date())
        if any_opt(log, ["some", "all"]): print(f"min_value = {min_value}")
        if any_opt(log, ["some", "all"]): print(f"pos = {pos}")
        if any_opt(log, ["some", "all"]): print(f"len(s) = {len(s)}")
        total_hist[pos:pos + len(s)] += s

    return pd.Series(total_hist[:-1], np.array(bins_total[:-2]))

def lowres_shots_series(shots, avg_days):
    result = {}
    for k in shots:
        s:pd.Series = shots[k]
        hist = dat.lower_resolution(s.values, avg_days)
        bins = np.arange(
                s.index[0],
                s.index[0] + dt.timedelta(days=len(hist)),
                dt.timedelta(days=1),
            )
        result[k] = pd.Series(hist, bins)
    return result

identity = lambda x: x

def get_sizes_and_durations(shots):
    sizes = []
    durations = []
    for k in shots:
        s = shots[k]
        size = s.sum()
        duration = len(s)
        sizes.append(size)
        durations.append(duration)
    return np.array(sizes), np.array(durations)

def estimate_a_b(
        shot_count,
        total_flow,
        sizes,
        durations,
        log: Set[Literal["none", "some", "all", "tqdm", "error"]]="none",
    ):
    #
    # Calculating values for the total flow
    #
    if any_opt(log, ["some", "all"]): print(sorted(total_flow, reverse=True)[0:10])
    if any_opt(log, ["some", "all"]): print(
        f"sum(hist_total > 0) = {sum(total_flow > 0)},",
        f"len(hist_total) = {len(total_flow)}")

    mean_totalrate = np.mean(total_flow)
    variance_totalrate = np.var(total_flow)

    if any_opt(log, ["some", "all"]): print(f"mean_totalrate = {mean_totalrate} [mean_totalrate = np.mean(total_flow)]")
    if any_opt(log, ["some", "all"]): print(f"variance_totalrate = {variance_totalrate} [variance_totalrate = np.var(total_flow)]")
    if any_opt(log, ["some", "all"]): print(f"len(hist_total) = {len(total_flow)}")
    
    count_flows = shot_count
    if any_opt(log, ["some", "all"]): print(f"count_flows = {count_flows}")

    flow_arrival_rate = count_flows/len(total_flow)
    if any_opt(log, ["some", "all"]): print(f"flow_arrival_rate = {flow_arrival_rate} [flow_arrival_rate = count_flows/len(total_flow)]")

    #
    # Aggregating values for individual shots
    #
    mean_size = np.mean(sizes)
    mean_duration = np.mean(durations)
    if any_opt(log, ["some", "all"]): print(f"mean_size = {mean_size}")
    if any_opt(log, ["some", "all"]): print(f"mean_duration = {mean_duration}")
    std_size = np.std(sizes)
    std_duration = np.std(durations)
    if any_opt(log, ["some", "all"]): print(f"std_size = {std_size}")
    if any_opt(log, ["some", "all"]): print(f"std_duration = {std_duration}")

    e_sn2_dn = np.mean(sizes*sizes/durations)
    if any_opt(log, ["some", "all"]): print(f"e_sn2_dn = {e_sn2_dn} [e_sn2_dn = np.mean(sizes*sizes/durations)]")

    model_mean = flow_arrival_rate*mean_size
    if any_opt(log, ["some", "all"]): print(f"model_mean = {model_mean} [model_mean = flow_arrival_rate*mean_size]")
    model_variance_factor = flow_arrival_rate*e_sn2_dn
    if any_opt(log, ["some", "all"]): print(f"model_variance_factor = {model_variance_factor} [model_variance_factor = flow_arrival_rate*e_sn2_dn]")
    if model_variance_factor == 0:
        return None
    k = variance_totalrate/model_variance_factor
    if any_opt(log, ["some", "all"]): print(f"k = {k} [k = variance_totalrate/model_variance_factor]")
    # if 0.99 < k < 1.0: k = 1.0

    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        b = k - 1.0 + (k**2.0 - k)**0.5
    if any_opt(log, ["some", "all"]): print(f"b = {b} [b = k - 1 + (k**2.0 - k)**0.5]")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = (b + 1)*(sizes/(np.power(durations, b + 1)))
    if any_opt(log, ["array"]): print(f"a = {a}")
    if any_opt(log, ["some", "all"]): print(f"mean(a) = {np.mean(a)}")

    return {
        "a_mean": np.mean(a),
        "b": b,
        "k": k,
        "mean_size": mean_size,
        "mean_duration": mean_duration,
        "e_sn2_dn": e_sn2_dn,
        "std_size": std_size,
        "std_duration": std_duration,
        "mean_totalrate": mean_totalrate,
        "model_mean": model_mean,
        "variance_totalrate": variance_totalrate,
        "model_variance_factor": model_variance_factor,
        "std_totalrate": (variance_totalrate)**0.5,
        "lambda": flow_arrival_rate,
        # "min_value": min_value_total,
        # "max_value": max_value_total,
        "count_flows": count_flows,
    }


def flow_cve_number_split_year_month(
            df,
            log,
            avg_items=1,
            remove_outliers=False,
            transformation=None,
        ):
    df = df[df["source"] != "securityfocus"]

    values_total = df["event_datetime"]
    min_date_total: np.datetime64 = min(values_total)
    max_date_total: np.datetime64 = max(values_total)

    shots = get_shots(df, "cve_number", "event_datetime")
    if avg_items > 1:
        shots = lowres_shots_series(shots, avg_items)
    if transformation is not None:
        shots = transform_shots_series(shots, transformation)

    if any_opt(log, ["some", "all"]): print(f"min_date_total = {min_date_total}")
    if any_opt(log, ["some", "all"]): print(f"max_date_total = {max_date_total}")

    intervals = dat.create_year_month_intervals(
        min_date_total.year, min_date_total.month,
        max_date_total.year, max_date_total.month,
    )

    rs = []
    from tqdm.auto import tqdm
    for year, month in tqdm(intervals):
        shots_ym, count = filter_shots_series_by_date(shots, dat.month_filter(year, month))
        if any_opt(log, ["some", "all"]): print(f"(year, month) = {(year, month)}")
        if any_opt(log, ["some", "all"]): print(f"len(shots_ym) = {len(shots_ym)}")
        if len(shots_ym) > 0:
            sizes, durations = get_sizes_and_durations(shots_ym)
            total_flow = get_aggregated_flow(shots_ym, log="none")
            r = estimate_a_b(count, total_flow, sizes, durations, log="none")
            r["year"] = year
            r["month"] = month
            rs.append(r)

    return pd.DataFrame(rs)

from tqdm import tqdm
def experiment_simulation__variance_grows_with_window(
            b = 0,
            experiment_count = 100,
            time_count = 100,
            mean_duration = 100,
            std_duration = 10,
            rate = 1,
            rng_seed = None,
            rng = None,
            pbar:tqdm = "auto",
            delta_days=10,
        ):
    result_stats = {}
    result_total_flows = {}
    from tqdm import tqdm
    from libs.utils import Memoize
    pbar = tqdm(total=experiment_count*time_count) if pbar == "auto" else pbar
    for i in range(experiment_count):
        if rng is None:
            rng = np.random.default_rng(*([] if rng_seed is None else [rng_seed]))
        min_value = dt.datetime(2000, 1, 1)
        max_value = dt.datetime(2010, 1, 1) + dt.timedelta(mean_duration*delta_days)
        bins = np.array([min_value + dt.timedelta(days=x)
                for x in range((max_value - min_value).days + 2)])
        shots = {}
        number = 0
        pos = 0.0
        while pos < len(bins) - 1:
            pos += rng.exponential(1/rate)
            duration = abs(int(rng.normal(mean_duration, std_duration))-1)+1
            int_pos = int(pos)
            index = bins[int_pos:int_pos + duration]
            if len(index) == 0:
                continue
            
            if b == 0: values = np.ones(len(index))
            elif b == 1: values = np.arange(1., len(index)+1)
            else: values = np.arange(1., len(index)+1)**b
            
            shot = pd.Series(values, index)
            shots[f"CVE-{number}"] = shot
            number += 1

        sub_result = []
        result_stats[f"sim{i}"] = sub_result
        result_total_flows[f"sim{i}"] = get_aggregated_flow(shots, log="none")
        for sz in range(1, time_count+1):
            filter = dat.date_interval_filter((2005, 1, 1), dt.timedelta(days=delta_days)*sz)
            Memoize.ignore = True
            shots2, count = filter_shots_series_by_date(
                shots, filter, cut_on_border=False)
            sizes, durations = get_sizes_and_durations(shots2)
            Memoize.ignore = False
            
            if len(shots2) > 0:
                total_flow = get_aggregated_flow(shots2, log="none")
                total_flow = filter_series_by_date_index(total_flow, filter)
                stats = estimate_a_b(count, total_flow, sizes, durations, log="none")
                sub_result.append(stats)
            else:
                sub_result.append(None)
            if pbar is not None: pbar.update(1)
            
    return result_stats, result_total_flows

import math
def rand_fair_product(rng=None, rand_seed=None, count=None, distributions=[]):
    # this implements Latin Hypercube with multiple finite countable distributions
    if rng is None:
        rng = np.random.default_rng(*([rand_seed] if rand_seed is not None else []))
    if count is None:
        count = math.lcm(*map(len, distributions))
    items = set()
    dist_values = [[]]*len(distributions)
    item = [None]*len(dist_values)
    for _ in range(count):
        while True:
            for i in range(len(dist_values)):
                if len(dist_values[i]) == 0:
                    dist_values[i] = [*distributions[i]]
                    rng.shuffle(dist_values[i])
                item[i] = dist_values[i].pop()
            item_to_add = (*item,)
            if item_to_add in items:
                for i in range(len(dist_values)):
                    dist_values[i].append(item_to_add[i])
                    rng.shuffle(dist_values[i])
                continue
            else:
                items.add(item_to_add)
                break
    return items

def do_experiment_simulation__variance_grows_with_window(args):
    md, sdp, r, b, experiment_count, time_count, rng_seed = args
    rng = np.random.default_rng(rng_seed)
    sd = md*sdp
    fname = f"result/experiment_with_growing_window_10to{time_count*10}days_nobordercut_b={b}_md={md}_sdp={sdp}_r={r}_{experiment_count}x.json"
    if os.path.isfile(fname):
        return
    exp0529_X, exp0529_X_flows = experiment_simulation__variance_grows_with_window(
            b=b,
            experiment_count=experiment_count,
            time_count=time_count, # x10 days (e.g. 200 means 2000 days)
            delta_days=10,
            mean_duration=md,
            std_duration=sd,
            rate=r,
            pbar=None,
            rng=rng,
        )
    df = pd.DataFrame(exp0529_X)
    df.to_json(fname, indent=2)

def do_many_experiments_simulation__variance_grows_with_window():
    from multiprocessing import Pool
    mds = [25, 100, 400]
    sdps = [.10, .25, .50]
    rs = [0.1, 0.4, 1., 2.]
    bs = [0, 1, 2, 3, 5]
    experiment_count = [100]
    time_count = [150] # x10 days (e.g. 200 means 2000 days)
    rng = np.random.default_rng(759874)
    rfp = rand_fair_product(rng=rng, distributions=[mds, sdps, rs, bs, experiment_count, time_count])
    from tqdm import tqdm
    # pbar = tqdm(total=experiment_count*time_count*len(rfp))

    with Pool(processes=16) as p:
        with tqdm(total=len(rfp)) as pbar:
            args = [(*x, rng.integers(9223372036854775807)) for x in rfp]
            for _ in p.imap_unordered(
                    do_experiment_simulation__variance_grows_with_window,
                    args,
                ):
                pbar.update()

def train_error_nn_model(df_error, random_state):
    df_error_X = df_error[["w", "b", "md", "sdp", "r"]]
    df_error_Y = df_error["value"]
    
    from sklearn.neural_network import MLPRegressor

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        df_error_X, df_error_Y, random_state=random_state)

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    
    max_score = -1e20
    model = None
    for alpha in [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1]:
        clf = MLPRegressor(alpha=alpha,
                            hidden_layer_sizes=(10, 10, 10, 10),
                            max_iter=1000,
                            random_state=random_state)
        from sklearn.model_selection import cross_val_score
        score = cross_val_score(clf, X_train, y_train, cv=5).mean()
        if score > max_score:
            max_score = score
            model = clf
    
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    
    from sklearn.metrics import mean_squared_error, mean_absolute_error
    rmse = mean_squared_error(y_test, pred)**0.5
    mae = mean_absolute_error(y_test, pred)
    
    return model, scaler, max_score, rmse, mae

def do_train_error_nn_model(args):
    return train_error_nn_model(*args)

def do_many_train_error_nn_model(df_error, count=100, random_state=759874):
    from multiprocessing import Pool
    rng = np.random.default_rng(random_state)
    model_infos = []
    from tqdm import tqdm
    with Pool(processes=16) as p:
        with tqdm(total=count) as pbar:
            args = [(df_error, rng.integers(2**32 - 1)) for _ in range(count)]
            for model_info in p.imap_unordered(
                    do_train_error_nn_model,
                    args,
                ):
                model, scaler, max_score, rmse, mae = model_info
                # print(f"max_score = {max_score}")
                # print(f"rmse = {rmse}")
                # print(f"mae = {mae}")
                pbar.update()
                model_infos.append(model_info)
    return model_infos

def get_outlier_fences(data, k=1.5):
    # Calculate the first and third quartiles
    q1 = np.percentile(data, 25)
    q3 = np.percentile(data, 75)
    
    # Calculate the interquartile range (IQR)
    iqr = q3 - q1
    
    # Define the lower and upper fences
    lower_fence = q1 - k * iqr
    upper_fence = q3 + k * iqr
    
    return lower_fence, upper_fence

def load_relative_errors_from_simulations(error_type="relative-datadriven-simulation"):
    from glob import glob
    import os
    import re
    from tqdm import tqdm
    dfs_errors = []
    for name in tqdm(glob("outputs/experiment_with_growing_window_10to1500days*.json")):
        # print(name)
        _, name = os.path.split(name)
        m = re.match(r"experiment_with_growing_window_10to(\d+)0days_nobordercut_b=(\d+)_md=(\d+)_sdp=([\d\.]+)_r=([\d\.]+)_(\d+)x.json", name)
        if m is not None:
            daysX10, b, md, sdp, r, count = int(m[1]), int(m[2]), int(m[3]), float(m[4]), float(m[5]), int(m[6])
            df0 = pd.read_json(f"outputs/{name}")
            df_var_pred = df0.applymap(lambda x: (((b+1)**2)/(2*b+1)) * x["model_variance_factor"] if x is not None else None)
            
            if error_type == "relative-datadriven-simulation":
                df_var_real = df0.applymap(lambda x: x["variance_totalrate"] if x is not None else None)
                df_error = (df_var_pred - df_var_real)/df_var_real
            if error_type == "relative-datadriven-model":
                s2_d = calculate_s2_d_ground_truth(md, sdp, b, "integral")
                analytic_variance = (((b+1)**2.0)/(2*b+1)) * r*s2_d
                df_error = (df_var_pred - analytic_variance)/analytic_variance
            if error_type == "absolute-datadriven-model":
                s2_d = calculate_s2_d_ground_truth(md, sdp, b, "integral")
                analytic_variance = (((b+1)**2.0)/(2*b+1)) * r*s2_d
                df_error = df_var_pred - analytic_variance

            df_error = pd.DataFrame({"value": df_error.T.mean()}).reset_index().rename(columns={"index": "w"})
            df_error["w"] = df_error["w"]*10 + 10
            df_error["b"] = b
            df_error["md"] = md
            df_error["sdp"] = sdp
            df_error["r"] = r
            df_error = df_error[["w", "b", "md", "sdp", "r", "value"]]
            dfs_errors.append(df_error)
    df_error = pd.concat(dfs_errors)

    # df_error.sort_values(by="value").to_csv("outputs/df_error_relative.csv")
    
    lower_fence, upper_fence = get_outlier_fences(df_error["value"])
    df_error = df_error[(lower_fence <= df_error["value"]) & (df_error["value"] <= upper_fence)]

    # df_error.sort_values(by="value").to_csv("outputs/df_error_relative_remoutlier.csv")/
    
    return df_error


def calculate_s2_d_ground_truth(md, sdp, b, method, int_range=10, sim_count=100_000):
    from statistics import NormalDist
    import math
    from libs.stats import expected_value, expected_value_sim
    dist = NormalDist(mu=md, sigma=md*sdp)
    if method == "wrong":
        s2_d = (sum(np.arange(1, md+1)**b))**2.0/md
    if method == "integral":
        std = math.floor(int_range*md*sdp)
        s2_d = expected_value(
                value_fn=lambda d:
                    (sum(np.arange(1., (abs(int(d)-1)+1)+1)**float(b)))**2.0 # Sn^2
                    /(abs(int(d)-1)+1) # /Dn
                    ,
                cdf_fn=dist.cdf,
                range=np.arange(md - std, md + std, 1),
            )
    if method == "simulation":
        s2_d = expected_value_sim(
                value_fn=lambda d:
                    (sum(np.arange(1., (abs(int(d)-1)+1)+1)**float(b)))**2.0 # Sn^2
                    /(abs(int(d)-1)+1) # /Dn
                    ,
                samples=dist.samples(sim_count),
            )
    return s2_d

def cache_pickle(func, fname):
    import pickle
    if not os.path.isfile(fname):
        obj = func()
        with open(fname, "bw") as fp:
            pickle.dump(obj, fp)
    else:
        with open(fname, "br") as fp:
            obj = pickle.load(fp)
    return obj
    

