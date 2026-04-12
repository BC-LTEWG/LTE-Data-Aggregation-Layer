import os
from .parameters import Params
from typing import Tuple
import numpy as np
import subprocess
import json
from scipy.linalg import inv
# models.LTE.simulation.parameters

# REPLACE WITH THE FILE PATH TO YOUR OWN BINARY. IF USING WINDOWS, MAKE SURE YOU USE DOUBLE FORWARD SLASHES (e.g. C:\\Users\\...)
EXE_PATH = "/home/alex/github/Agent-Based-Simulation-Model/Agent-Based Simulation/bin/sim" # sal/me
# EXE_PATH = "/home/alex/github/another/Agent-Based-Simulation-Model/Agent-Based Simulation/bin/sim" # devin


def get_theoretical_values(A, l):
    n = A.shape[0]
    vals = inv(np.eye(n) - A.T)@l

    return vals

def get_supply(n_products, dist_invs, prod_invs= None):
    supply = np.zeros(n_products)

    for _, inventory in dist_invs.items():
        supply += inventory

    if prod_invs != None:
        for _, inventory in prod_invs.items():
            supply += inventory

    return supply

def _get_average_endowments(persons, n_commodities):
    endowments = [dic["endowment"] for _,dic in persons.items()]
    itemwise_endowments = [[] for i in range(n_commodities)]
    for i in range(n_commodities):
        for j in range(len(persons)):
            itemwise_endowments[i].append(endowments[j][i])

    average_endowments = [np.average(itemwise_endowments[i]) for i in range(n_commodities)]
    return average_endowments

def _get_average_abilities(persons, n_abilities):
    abilities = [dic["abilities"] for _,dic in persons.items()]

    abilitywise_profs = [[] for i in range(n_abilities)]
    for i in range(n_abilities):
        for j in range(len(persons)):
            abilitywise_profs[i].append(abilities[j][i])

    average_proficiencies = [np.average(abilitywise_profs[i]) for i in range(n_abilities)]
    return average_proficiencies

def _get_average_demand(producer_dict, distributor_dict, n_prods):
    average_demands = np.zeros(n_prods)
    for j in range(n_prods):
        all_demands_producer = [producer_dict[i][j] for i in producer_dict]
        all_demands_distributor = [distributor_dict[i][j] for i in distributor_dict]
        all_demands = all_demands_producer + all_demands_distributor
        average_demands[j] = np.average(all_demands)

    return average_demands

def update_hourly_stats(
        traj, prices, persons,
        producer_inventories, distributor_inventories,
        active_plans, n_products, n_commodities, n_abilities,
        current_employment, t,
        producer_inventories_micro, distributor_inventories_micro,
        consumption_frequencies, consumption_periods,
        producer_demand_signals, distributor_demand_signals,
        producer_pending_inventories, distributor_pending_inventories,
):

    overall_supply = get_supply(n_products, distributor_inventories, producer_inventories)
    accessible_supply = get_supply(n_products, distributor_inventories)

    overall_supply_micro = get_supply(n_products, distributor_inventories_micro, producer_inventories_micro)
    accessible_supply_micro = get_supply(n_products, distributor_inventories_micro)

    average_demands = _get_average_demand(producer_demand_signals, distributor_demand_signals, n_products)
    average_pending_inventories = _get_average_demand(producer_pending_inventories, distributor_pending_inventories, n_products)

    accounts = [dic["account"] for _, dic in persons.items()]
    health_statuses = [0 if dic["health"] == "Healthy" else 1 for _, dic in persons.items()]
    n_unhealthy = sum(health_statuses)
    n_healthy = len(persons) - n_unhealthy

    average_endowments = _get_average_endowments(persons, n_commodities)
    average_proficiencies = _get_average_abilities(persons, n_abilities)

    plans_in_motion = [active_plans[i]["plans"] for i in range(n_products)]
    quantities_in_prod = [active_plans[i]["quantity"] for i in range(n_products)]

    traj["prices"] = np.append(traj["prices"], [prices], axis= 0)
    traj["values"] = np.append(traj["values"], [traj["values"][-1]], axis= 0)
    traj["supply"] = np.append(traj["supply"], [overall_supply], axis=0)
    traj["supply_micro"] = np.append(traj["supply_micro"], [overall_supply_micro], axis=0)
    traj["accessible_supply"] = np.append(traj["accessible_supply"], [accessible_supply], axis= 0)
    traj["accessible_supply_micro"] = np.append(traj["accessible_supply_micro"], [accessible_supply_micro], axis= 0)
    traj["avg_account"] = np.append(traj["avg_account"], np.average(accounts))
    traj["avg_endowments"] = np.append(traj["avg_endowments"], [average_endowments], axis= 0)
    traj["plans_in_progress"] = np.append(traj["plans_in_progress"], [plans_in_motion], axis= 0)
    traj["goods_in_production"] = np.append(traj["goods_in_production"], [quantities_in_prod], axis= 0)
    traj["n_healthy"] = np.append(traj["n_healthy"], n_healthy)
    traj["n_unhealthy"] = np.append(traj["n_unhealthy"], n_unhealthy)
    traj["average_proficiencies"] = np.append(traj["average_proficiencies"], [average_proficiencies], axis= 0)
    traj["employment"] = np.append(traj["employment"], current_employment)
    traj["mean_consumption_frequencies"] = np.append(traj["mean_consumption_frequencies"], [consumption_frequencies], axis= 0)
    traj["mean_consumption_periods"] = np.append(traj["mean_consumption_periods"], [consumption_periods], axis= 0)
    traj["average_demand"] = np.append(traj["average_demand"], [average_demands], axis= 0)
    traj["average_pending_inventories"] = np.append(traj["average_pending_inventories"], [average_pending_inventories], axis= 0)

def update_daily_stats():
    pass

def get_trajectories(params: Params):

    n_commodities = params.N_c
    n_time_steps = params.N_S
    num_persons = params.N_h
    prods_per_machine = params.m_r
    n_producers = params.N_p
    n_distributors = params.N_c
    init_working_day = params.T
    init_working_week = params.W
    n_abilities = 3 # needs to be a parameter

    reasonable_logs = n_commodities < 10 and num_persons < 50 and n_time_steps < 10000
    print(f"{reasonable_logs=}")

    args = [
        "-j",
        "-n", str(n_time_steps),
        "-p", str(num_persons),
        "-h", str(init_working_day),
        "-w", str(init_working_week),
        "-o", str(n_commodities),
        "-m", str(prods_per_machine),
        "-r", str(n_producers),
        "-d", str(n_distributors),
    ]

    if os.name == "nt":
        proc = subprocess.Popen(
            ["cmd", "/c", EXE_PATH, *args],
            stdout= subprocess.PIPE,
            stderr= subprocess.PIPE,
            creationflags=subprocess.CREATE_NO_WINDOW,
            text= True,
            bufsize= 1,
        )
    else:
        proc = subprocess.Popen(
            [EXE_PATH, *args],
            stdout= subprocess.PIPE,
            stderr= subprocess.PIPE,
            text= True,
            bufsize= 1,
        )
    assert proc.stdout is not None

    n_prods = n_commodities + n_commodities // 5 # 1 machine per 5 products right now is hard-wired
    prices = np.zeros(n_prods)
    consumption_frequencies = np.zeros(n_prods)
    consumption_periods = np.zeros(n_prods)
    persons = {i: {
        "account": 0,
        "endowment": np.zeros(n_commodities),
        "abilities": np.zeros(n_abilities),
        "health": "Healthy" # everyone starts in good health
    } for i in range(num_persons)}
    producer_inventories = {i: np.zeros(n_prods) for i in range(n_producers)}
    distributor_inventories = {i: np.zeros(n_prods) for i in range(n_distributors)}
    producer_inventories_micro = {i: np.zeros(n_prods) for i in range(n_producers)}
    distributor_inventories_micro = {i: np.zeros(n_prods) for i in range(n_distributors)}
    firm_inventories = {i: np.zeros(n_commodities) for i in range(n_producers + n_distributors)}
    active_plans = {i: {"plans": 0, "quantity": 0} for i in range(n_prods)}
    producer_demand_signals = {i: np.zeros(n_prods) for i in range(n_producers)}
    distributor_demand_signals = {i: np.zeros(n_prods) for i in range(n_distributors)}
    producer_pending_inventories = {i: np.zeros(n_prods) for i in range(n_producers)}
    distributor_pending_inventories = {i: np.zeros(n_prods) for i in range(n_distributors)}
    order_sizes = {}
    current_t = 0
    t = [0]

    A = np.zeros((n_prods, n_prods))
    l = np.zeros(n_prods)

    print(f"{os.getcwd()}")
    # with open("/media/Big-Boy/Nextcloud/Personal-Programming/python/Modeling-Tools-Data/models/LTE/output_log.txt", "w") as f:

        # initialization
    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        try:
            dic = json.loads(line)
            # if reasonable_logs:
            #     print(dic, file= f)
            if dic["t"] != 0:
                break
            id = dic["id"]
            client = dic.get("client", "")
            label = dic.get("label", "")
            values = dic.get("values", [])
            match client:
                case "Society":
                    if label == "A":
                        coords = tuple(dic.get("coords"))
                        i, j = coords
                        a_ij = dic.get("value")
                        A[i][j] = a_ij
                    if label == "l":
                        i = dic["index"]
                        value = dic["value"]
                        l_i = value
                        l[i] = l_i
                    if label == "price":
                        prices[id] = values[0]
                    if label == "mean_consumption_frequency":
                        pair = list(dic.items())[-1]
                        prod_str = pair[0]
                        prod_id = int(prod_str.split('_')[1])
                        val = pair[1]
                        consumption_frequencies[prod_id] = val
                    if label == "mean_consumption_period":
                        pair = list(dic.items())[-1]
                        prod_str = pair[0]
                        prod_id = int(prod_str.split('_')[1])
                        val = pair[1]
                        consumption_periods[prod_id] = val

                case "Person":
                    if label == "age":
                        persons[id]["health"] = values[0]
                    if label == "account":
                        persons[id]["account"] = values[0]
                    if label == "health_status":
                        persons[id]["health"] = values[0]
                    if label == "consumption":
                        pair = list(dic.items())[-1]
                        prod_str = pair[0]
                        prod_id = int(prod_str.split('_')[1])
                        amt = pair[1]
                        persons[id]["endowment"][prod_id] -= amt
                    if label == "ability":
                        ability_id = dic["ability"]
                        val = dic["value"]
                        person_id = dic['id']
                        persons[person_id]["abilities"][ability_id-1] = val
                    if label == "inventory":
                        prod_id = dic["product_id"]
                        amt = dic["amount"]
                        person_id = dic['id']
                        persons[person_id]["endowment"][prod_id] = amt
                    if label == "purchase":
                        pair = list(dic.items())[-1]
                        prod_str = pair[0]
                        prod_id = int(prod_str.split('_')[1])
                        amt = pair[1]
                        persons[id]["endowment"][prod_id] += amt

                        cost = prices[prod_id]*amt
                        persons[id]["account"] -= cost

                case "Firm":
                    if label == "initial_inventory":
                        temp = values[0].split("_")
                        prod = int(temp[1])
                        quant = values[1]
                        firm_inventories[id][prod] = quant

                case "Producer":
                    if label == "inventory_level":
                        prod_id = dic["product_id"]
                        amt = dic["amount"]
                        producer_id = dic['id']
                        producer_inventories[producer_id][prod_id] = amt
                        producer_inventories_micro[producer_id][prod_id] = amt

                    if label == "inventory_reduction":
                        prod_id = dic["product_id"]
                        amt = dic["amount"]
                        # distributor_inventories[id-n_producers][prod_id] -= amt
                        producer_inventories_micro[id][prod_id] -= amt

                case "Distributor":
                    if label == "inventory_level":
                        prod_id = dic["product_id"]
                        amt = dic["amount"]
                        distributor_id = dic['id']
                        distributor_inventories[distributor_id-n_producers][prod_id] = amt
                        distributor_inventories_micro[distributor_id-n_producers][prod_id] = amt

                    if label == "inventory_reduction":
                        prod_id = dic["product_id"]
                        amt = dic["amount"]
                        # distributor_inventories[id-n_producers][prod_id] -= amt
                        distributor_inventories_micro[id-n_producers][prod_id] -= amt

        except json.decoder.JSONDecodeError:
            print(f"NON-JSON RUNTIME LINE: {line!r}")
            continue

    turnover_times = [[] for n in range(n_prods)]
    current_employment = 0

    traj = {
        "prices": np.array([prices]),
        "values": np.array([prices]),
        "theoretical_values": np.array([get_theoretical_values(A,l)]),
        "supply": np.array([get_supply(n_prods, distributor_inventories, producer_inventories)]),
        "supply_micro": np.array([get_supply(n_prods, distributor_inventories_micro, producer_inventories_micro)]),
        "accessible_supply": np.array([get_supply(n_prods, distributor_inventories)]),
        "accessible_supply_micro": np.array([get_supply(n_prods, distributor_inventories_micro)]),
        "avg_account": np.array([np.average([dic["account"] for _, dic in persons.items()])]),
        "avg_turnover_times": np.array([turnover_times]),
        "avg_endowments": np.array([_get_average_endowments(persons, n_commodities)]),
        "A": A,
        "plans_in_progress": np.array([np.zeros(n_prods)]),
        "goods_in_production": np.array([np.zeros(n_prods)]),
        "n_healthy": np.array([num_persons]),
        "n_unhealthy": np.array([0]),
        "average_proficiencies": np.array([_get_average_abilities(persons, n_abilities)]),
        "employment": np.array([current_employment]),
        "mean_consumption_frequencies": np.array([consumption_frequencies]),
        "mean_consumption_periods": np.array([consumption_periods]),
        "average_demand": np.array([np.zeros(n_prods)]),
        "average_pending_inventories": np.array([np.zeros(n_prods)])
    }

    yield traj, t

    # with open("/media/Big-Boy/Nextcloud/Personal-Programming/python/Modeling-Tools-Data/models/LTE/output_log.txt", "a") as f:

    for line in proc.stdout:
        line = line.strip()
        if not line:
            continue

        dic = json.loads(line)
        # if reasonable_logs:
        #     print(dic, file= f)
        if dic["t"] != current_t:
            current_t = dic["t"]
            t.append(current_t)
            update_hourly_stats(
                traj, prices, persons, 
                producer_inventories, distributor_inventories,
                active_plans, n_prods, n_commodities, n_abilities,
                current_employment, t,
                producer_inventories_micro, distributor_inventories_micro,
                consumption_frequencies, consumption_periods,
                producer_demand_signals, distributor_demand_signals,
                producer_pending_inventories, distributor_pending_inventories
            )
            yield traj, t

            # if current_t % 24 == 0 and current_t != 0:
            #     update_daily_stats(traj)
            
        # yield traj, t
        id = dic["id"]
        client = dic.get("client", "")
        label = dic.get("label", "")
        values = dic.get("values", [])

        match client:
            case "Person":
                if label == "purchase":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]
                    persons[id]["endowment"][prod_id] += amt

                if label == "hours":
                    persons[id]["account"] += values[0]

                if label == "consumption":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]
                    persons[id]["endowment"][prod_id] -= amt

                if label == "health_status":
                    persons[id]["health"] = values[0]

                if label == "account":
                    if label == "account":
                        persons[id]["account"] = values[0]

            case "Distributor":
                if label == "accepted_order":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    time = pair[1]
                    turnover_times[prod_id].append(time)

                if label == "inventory_level":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    distributor_inventories[id-n_producers][prod_id] = amt

                if label == "shipment_received":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    distributor_inventories_micro[id-n_producers][prod_id] += amt

                if label == "inventory_reduction":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    distributor_inventories_micro[id-n_producers][prod_id] -= amt

                if label == "current_demand":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    demand = pair[1]
                    distributor_demand_signals[id-n_producers][prod_id] = demand

                if label == "pending_inventory":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    threshold = pair[1]
                    distributor_pending_inventories[id-n_producers][prod_id] = threshold

            case "Producer":
                if label == "pursued_plan":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]

                    active_plans[prod_id]["plans"] += 1
                    active_plans[prod_id]["quantity"] += amt

                if label == "ended_plan":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]

                    active_plans[prod_id]["plans"] -= 1
                    active_plans[prod_id]["quantity"] -= amt

                if label == "shipment_received":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    producer_inventories_micro[id][prod_id] += amt

                if label == "inventory_reduction":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    producer_inventories_micro[id][prod_id] -= amt

                if label == "inventory_level":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    producer_inventories[id][prod_id] = amt

                if label == "current_demand":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    demand = pair[1]
                    producer_demand_signals[id][prod_id] = demand

                if label == "pending_inventory":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    threshold = pair[1]
                    producer_pending_inventories[id][prod_id] = threshold

            case "Society":
                if label == "price":
                    prices[id] = values[0]

                if label == "employment":
                    current_employment = dic["values"][0]
                    
