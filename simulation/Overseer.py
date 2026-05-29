import numpy as np
from .Collector import Collector
from typing import Tuple
from scipy.linalg import inv
from pathlib import Path
import math
import copy
from overseer.tools.dataclasses import Replace, Extend, Append

class Overseer:
    """ The guy who watches the LTE and keeps track of the data. """
    def __init__(self, bin_path, params, log_path_str: str | None = None):
        self.params = params

        if log_path_str is not None:
            log_path = Path(log_path_str).expanduser().resolve()
            log_filename = log_path.name
            log_parent_dir = log_path.parent
            if log_parent_dir.exists():
                self.log_path = log_path
                with open(log_path, "w") as f:
                    pass
            else:
                print(f"{str(log_parent_dir)} does not exist.")
                self.log_path = None
        else:
            self.log_path = None


        # create a more human readable dictionary of helpful quantities
        self.settings = {
            "n_commodities": params.N_c,
            "n_time_steps": params.N_S,
            "n_persons": params.N_h,
            "prods_per_machine": params.m_r,
            "n_producers": params.N_p,
            "n_distributors": params.N_c,
            "init_working_day": params.T,
            "init_working_week": params.W,
            "daily_sick_chance": params.S,
            "person_ability_stddev": params.v_ability,
            "n_abilities": 3, # needs to be a parameter
            # the rest of these are dependent variables
            "n_products": 2 * params.N_c + params.N_c // params.m_r,
        }

        reasonable_logs = params.N_c < 10 and params.N_S <= 3000 and params.N_h < 100
        self.is_logging = params.is_logging and reasonable_logs and self.log_path is not None
        self.is_logging = False

        # create and start the collection thread
        args = self._get_args_from_settings()
        self.collector = Collector(bin_path, args)
        self.collector.start_sim_and_begin_collection()

        # internal stuff
        self.text_log = []
        self.t = [0]
        self.current_t = 0
        self.stdout_done = False
        self.stderr_done = False

        # HERE IS WHERE YOU WOULD DECLARE ANY QUANTITIES WHICH YOU WANT THE OVERSEER TO KEEP TRACK OF
        self.prices = np.zeros(self.settings["n_products"])
        self.turnover_times = [[] for n in range(self.settings["n_products"])]
        self.current_employment = 0

        self.persons = {i: {
            "account": 0,
            "endowment": np.zeros(self.settings["n_commodities"]),
            "abilities": np.zeros(self.settings["n_abilities"]),
            "health": "Healthy", # everyone starts in good health
            "recent_busyness": 0.0
        } for i in range(self.settings["n_persons"])}

        self.producers = {i: {
            "employees": 0,
            "inventory": np.zeros(self.settings["n_products"]),
            "inventory_micro": np.zeros(self.settings["n_products"]),
            "pending_inventory": np.zeros(self.settings["n_products"]),
            "demand_signals": np.zeros(self.settings["n_products"]),
            "catalog": [],
            "recent_busyness": 0,
            "recent_weekly_busyness": 0,
            "inc_inventory": np.zeros(self.settings["n_products"])
        } for i in range(self.settings["n_producers"])}

        self.distributors = {i: {
            "employees": 0,
            "inventory": np.zeros(self.settings["n_products"]),
            "inventory_micro": np.zeros(self.settings["n_products"]),
            "pending_inventory": np.zeros(self.settings["n_products"]),
            "demand_signals": np.zeros(self.settings["n_products"]),
            "catalog": [],
            "recent_busyness": 0,
            "recent_weekly_busyness": 0,
            "inc_inventory": np.zeros(self.settings["n_products"])
        } for i in range(self.settings["n_distributors"])}


        self.A = np.zeros((self.settings["n_products"], self.settings["n_products"]))
        self.l = np.zeros(self.settings["n_products"])
        self.consumption_frequencies = np.zeros(self.settings["n_products"])
        self.consumption_periods = np.zeros(self.settings["n_products"])
        self.order_sizes = [[] for i in range(self.settings["n_products"])]
        self.transfer_requests_by_sector = np.zeros(self.settings["n_products"])
        self.transfer_requests_by_sector_t = np.array([])
        self.active_plans = {i: {"plans": 0, "quantity": 0, "actual_quantity": 0} for i in range(self.settings["n_products"])}
        self.reorder_requests = np.zeros(self.settings["n_products"])
        self.overall_busyness = 0
        self.overall_weekly_busyness = 0
        self.long_run_employment_by_sector = np.zeros(self.settings["n_products"])
        self.long_run_sector_activity = np.zeros(self.settings["n_products"])
        self.long_run_actual_sector_activity = np.zeros(self.settings["n_products"])

    def _process_dic(self, dic):
        """ 
        Looks at the contents of a json logged dictionary and updates the relevant quantities accordingly
        """
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
                    self.A[i][j] = a_ij

                if label == "l":
                    i = dic["index"]
                    value = dic["value"]
                    l_i = value
                    self.l[i] = l_i

                if label == "price":
                    self.prices[id] = values[0]

                if label == "mean_consumption_frequency":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    val = pair[1]
                    self.consumption_frequencies[prod_id] = val

                if label == "mean_consumption_period":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    val = pair[1]
                    self.consumption_periods[prod_id] = val

                if label == "employment":
                    self.current_employment = dic["values"][0]

            case "Person":
                if label == "age":
                    self.persons[id]["health"] = values[0]

                if label == "account":
                    self.persons[id]["account"] = values[0]

                if label == "health_status":
                    self.persons[id]["health"] = values[0]

                if label == "consumption":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]
                    self.persons[id]["endowment"][prod_id] = amt

                if label == "ability":
                    ability_id = dic["ability"]
                    val = dic["value"]
                    person_id = dic['id']
                    self.persons[person_id]["abilities"][ability_id-1] = val

                if label == "inventory":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    person_id = dic['id']
                    self.persons[person_id]["endowment"][prod_id] = amt

                if label == "purchase":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]
                    self.persons[id]["endowment"][prod_id] += amt
                    cost = self.prices[prod_id]*amt
                    self.persons[id]["account"] -= cost

                if label == "hours":
                    self.persons[id]["account"] += values[0]


            case "Producer":
                if label == "inventory_level":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    producer_id = dic['id']
                    self.producers[producer_id]["inventory"][prod_id] = amt
                    self.producers[producer_id]["inventory_micro"][prod_id] = amt

                if label == "inventory_reduction":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    # distributor_inventories[id-n_producers][prod_id] -= amt
                    self.producers[id]["inventory_micro"][prod_id] -= amt

                if label == "catalog":
                    self.producers[id]["catalog"] = values

                if label == "pursued_plan":
                    producer_id = id
                    customer_id = values[0]
                    is_distributor = (customer_id >= self.settings["n_producers"])
                    if is_distributor:
                        customer_id = self._get_dist_key(customer_id)
                    prod_id = values[1]
                    amt = values[2]
                    n_workers = values[3]
                    actual_amt = self.get_expected_quantity(prod_id, amt, n_workers)

                    self.active_plans[prod_id]["plans"] += 1
                    self.active_plans[prod_id]["quantity"] += amt
                    self.active_plans[prod_id]["actual_quantity"] += actual_amt
                    self.long_run_sector_activity[prod_id] += amt
                    self.long_run_actual_sector_activity[prod_id] += actual_amt

                    if is_distributor:
                        self.distributors[customer_id]["inc_inventory"][prod_id] += amt
                    else:
                        self.producers[customer_id]["inc_inventory"][prod_id] += amt

                # if label == "pursued_plan":
                #     producer_id = id
                #     customer_id = values[0]
                #     is_distributor = (customer_id >= self.settings["n_producers"])
                #     if is_distributor:
                #         customer_id = self._get_dist_key(customer_id)
                #     prod_id = values[1]
                #     amt = values[2]

                #     self.active_plans[prod_id]["plans"] += 1
                #     self.active_plans[prod_id]["quantity"] += amt
                #     if is_distributor:
                #         self.distributors[customer_id]["inc_inventory"][prod_id] += amt
                #     else:
                #         self.producers[customer_id]["inc_inventory"][prod_id] += amt

                if label == "ended_plan":
                    prod_str = values[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = values[1]
                    actual_amt = values[2]

                    # pair = list(dic.items())[-1]
                    # prod_str = pair[0]
                    # prod_id = int(prod_str.split('_')[1])
                    # amt = pair[1]
                    self.active_plans[prod_id]["plans"] -= 1
                    self.active_plans[prod_id]["quantity"] -= amt
                    self.active_plans[prod_id]["actual_quantity"] -= actual_amt

                if label == "shipment_received":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    self.producers[id]["inc_inventory"][prod_id] -= amt

                if label == "current_demand":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    demand = pair[1]
                    self.producers[id]["demand_signals"][prod_id] = demand

                # if label == "pending_inventory":
                #     pair = list(dic.items())[-1]
                #     prod_str = pair[0]
                #     prod_id = int(prod_str.split('_')[1])
                #     threshold = pair[1]
                #     self.producers[id]["pending_inventory"][prod_id] = threshold

                if label in {"reorder" "reorder_failure"}:
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    self.reorder_requests[prod_id] += 1

                if label == "newly employed":
                    prod_id = values[0]
                    self.producers[prod_id]["employees"] += 1

                if label == "busyness":
                    firm_busyness = values[0]
                    overall_busyness = values[1]
                    transfers_available = values[2]
                    self.producers[id]["recent_busyness"] = firm_busyness
                    self.overall_busyness = overall_busyness

                if label == "weekly_busyness":
                    firm_busyness = values[0]
                    overall_busyness = values[1]
                    transfers_available = values[2]
                    self.producers[id]["recent_weekly_busyness"] = firm_busyness
                    self.overall_weekly_busyness = overall_busyness

                if label == "accepted_order":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]
                    self.order_sizes[prod_id].append(amt)

                if label == "transfer_request":
                    cat = self.producers[id]["catalog"]
                    for i in cat:
                        self.transfer_requests_by_sector[i] +=  1
                        if len(self.transfer_requests_by_sector_t) == 0 or self.current_t != self.transfer_requests_by_sector_t[-1]:
                            self.transfer_requests_by_sector_t = np.append(self.transfer_requests_by_sector_t, self.current_t)

                if label == "transfer":
                    worker_id = id
                    old_emp = values[0]
                    new_emp = values[1]
                    self.producers[old_emp]["employees"] -= 1
                    self.producers[new_emp]["employees"] += 1

            case "Distributor":
                if label == "inventory_level":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    distributor_id = dic['id']
                    dist_key = self._get_dist_key(distributor_id)
                    self.distributors[dist_key]["inventory"][prod_id] = amt
                    self.distributors[dist_key]["inventory_micro"][prod_id] = amt

                if label == "inventory_reduction":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    dist_key = self._get_dist_key(id)
                    self.distributors[dist_key]["inventory_micro"][prod_id] -= amt

                if label == "catalog":
                    dist_key = self._get_dist_key(id)
                    self.distributors[dist_key]["catalog"] = values

                if label == "accepted_order":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    time = pair[1]
                    self.turnover_times[prod_id].append(time)

                if label == "shipment_received":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["inc_inventory"][prod_id] -= amt
                    self.distributors[dist_id]["inventory_micro"][prod_id] += amt

                if label == "current_demand":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    demand = pair[1]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["demand_signals"][prod_id] = demand

                # if label == "pending_inventory":
                #     pair = list(dic.items())[-1]
                #     prod_str = pair[0]
                #     prod_id = int(prod_str.split('_')[1])
                #     threshold = pair[1]
                #     dist_id = self._get_dist_key(id)
                #     self.distributors[dist_id]["pending_inventory"][prod_id] = threshold

                if label in {"reorder", "reorder_failure"}:
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    self.reorder_requests[prod_id] += 1

                if label == "newly employed":
                    dist_id = values[0]
                    dist_id = self._get_dist_key(dist_id)
                    self.distributors[dist_id]["employees"] += 1

                if label == "busyness":
                    firm_busyness = values[0]
                    overall_busyness = values[1]
                    transfers_available = values[2]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["recent_busyness"] = firm_busyness
                    self.overall_busyness = overall_busyness

                if label == "weekly_busyness":
                    firm_busyness = values[0]
                    overall_busyness = values[1]
                    transfers_available = values[2]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["recent_weekly_busyness"] = firm_busyness
                    self.overall_weekly_busyness = overall_busyness

                if label == "accepted_order":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]
                    self.order_sizes[prod_id].append(amt)

                if label == "transfer_request":
                    dist_id = self._get_dist_key(id)
                    cat = self.distributors[dist_id]["catalog"]
                    for i in cat:
                        self.transfer_requests_by_sector[i] +=  1
                        if len(self.transfer_requests_by_sector_t) == 0 or self.current_t != self.transfer_requests_by_sector_t[-1]:
                            self.transfer_requests_by_sector_t = np.append(self.transfer_requests_by_sector_t, self.current_t)

    def _update_hourly_stats(self):
        """ 
        Updates the trajectories dictionary.
        """
        overall_supply = self._get_supply(self.distributors, self.producers)
        accessible_supply = self._get_supply(self.distributors)

        overall_supply_micro = self._get_supply(self.distributors, self.producers, micro= True)
        accessible_supply_micro = self._get_supply(self.distributors, micro= True)

        average_demands = self._get_average(self.producers, self.distributors, key= "demand_signals")
        average_demands_producers = self._get_average(self.producers, key= "demand_signals")
        average_demands_distributors = self._get_average(self.distributors, key= "demand_signals")

        self._set_pending_inventories()
        average_pending_inventories = self._get_average(self.producers, self.distributors, key= "pending_inventory")
        average_pending_inventories_producers = self._get_average(self.producers, key= "pending_inventory")
        average_pending_inventories_distributors = self._get_average(self.distributors, key= "pending_inventory")

        accounts = [dic["account"] for _, dic in self.persons.items()]
        health_statuses = [0 if dic["health"] == "Healthy" else 1 for _, dic in self.persons.items()]
        n_unhealthy = sum(health_statuses)
        n_healthy = len(self.persons) - n_unhealthy

        average_endowments = self._get_average_endowments(self.persons)
        average_proficiencies = self._get_average_abilities(self.persons)

        plans_in_motion = [self.active_plans[i]["plans"] for i in range(self.settings["n_products"])]
        quantities_in_prod = [self.active_plans[i]["quantity"] for i in range(self.settings["n_products"])]
        actual_quantities_in_prod = [self.active_plans[i]["actual_quantity"] for i in range(self.settings["n_products"])]

        sectoral_employment = self._get_available_employment_by_sector(self.producers)
        sectoral_busyness = self._get_sectoral_busyness(self.producers)
        sectoral_weekly_busyness = self._get_sectoral_busyness(self.producers, weekly= True)

        order_size_averages = np.array([np.average(orders) for orders in self.order_sizes])

        self.long_run_employment_by_sector += sectoral_employment

        self.traj["prices"] = Append(self.prices)
        # self._update_data("prices", self.prices)
        self.traj["values"] = Append(self.values)
        # self._update_data("values", self.traj["values"][-1])
        self.traj["supply"] = Append(overall_supply)
        # self._update_data("supply", overall_supply)
        self.traj["supply_micro"] = Append(overall_supply_micro)
        self.traj["accessible_supply"] = Append(accessible_supply)
        self.traj["accessible_supply_micro"] = Append(accessible_supply_micro)
        self.traj["avg_account"] = Append(np.average(accounts))
        self.traj["min_account"] = Append(np.min(accounts))
        self.traj["max_account"] = Append(np.max(accounts))
        self.traj["avg_endowments"] = Append(average_endowments)
        self.traj["plans_in_progress"] = Append(plans_in_motion)
        self.traj["goods_in_production"] = Append(quantities_in_prod)
        self.traj["actual_goods_in_production"] = Append(actual_quantities_in_prod)
        self.traj["n_healthy"] = Append(n_healthy)
        self.traj["n_unhealthy"] = Append(n_unhealthy)
        self.traj["average_proficiencies"] = Append(average_proficiencies)
        self.traj["employment"] = Append(self.current_employment)
        self.traj["mean_consumption_frequencies"] = Append(self.consumption_frequencies)
        self.traj["mean_consumption_periods"] = Append(self.consumption_periods)
        self.traj["average_demand"] = Append(average_demands)
        self.traj["average_demand_producers"] = Append(average_demands_producers)
        self.traj["average_demand_distributors"] = Append(average_demands_distributors)
        self.traj["average_pending_inventories"] = Append(average_pending_inventories)
        self.traj["average_pending_inventories_distributors"] = Append(average_pending_inventories_distributors)
        self.traj["average_pending_inventories_producers"] = Append(average_pending_inventories_producers)
        self.traj["reorder_requests"] = Append(self.reorder_requests)
        self.traj["available_employment_by_sector"] = Append(sectoral_employment)
        self.traj["sectoral_busyness"] = Append(sectoral_busyness)
        self.traj["overall_busyness"] = Append(self.overall_busyness)
        self.traj["sectoral_weekly_busyness"] = Append(sectoral_weekly_busyness)
        self.traj["overall_weekly_busyness"] = Append(self.overall_weekly_busyness)
        self.traj["order_sizes"] = Append(order_size_averages)
        self.traj["l"] = Append(self.l)
        self.traj["transfer_requests_by_sector"] = Append(self.transfer_requests_by_sector)
        self.traj["long_run_employment_by_sector"] = Append(self.long_run_employment_by_sector / max(self.current_t, 1))
        self.traj["eqb_employment"] = Append(self.eqb_employment)
        self.traj["min_hrly_output"] = Append(self.min_hrly_output)
        self.traj["busy_lower_bound"] = Append(self.busy_lower_bd)
        self.traj["busy_upper_bound"] = Append(self.busy_upper_bd)
        self.traj["long_run_activity"] = Append(self.long_run_sector_activity / max(self.current_t, 1))
        self.traj["long_run_actual_activity"] = Append(self.long_run_actual_sector_activity / max(self.current_t, 1))
        self.traj["transfer_requests_by_sector_t"] = Replace(self.transfer_requests_by_sector_t)

        self.transfer_requests_by_sector = np.zeros(self.settings["n_products"])
        self.reorder_requests = np.zeros(self.settings["n_products"])

    def initialize_properties(self):
        N = self.settings["n_persons"]
        net_weekly_demand = N*24*7*self.consumption_frequencies
        gross_weekly_demand = inv(np.eye(self.settings["n_products"]) - self.A)@net_weekly_demand
        sectoral_weekly_labor_req = self.l * gross_weekly_demand
        min_hrly_output = gross_weekly_demand / (24*7)
        # gross_hrly_consumption = inv(np.eye(self.settings["n_products"])-self.A)@self.consumption_frequencies
        # omega = 8*5 / (24*7)

        self.eqb_employment = sectoral_weekly_labor_req / (8*5)
        self.busy_lower_bd = 0.7*(8*5 / (24*7))
        self.busy_upper_bd = (8*5 / (24*7))
        self.min_hrly_output = min_hrly_output
        self.values = copy.deepcopy(self.prices)

    def _declare_traj(self):
        """ 
        After time = 0 finishes, the trajectories dictionary is initialized from this function. You would need to add new ones to this if making your own.
        """
        return {
            "prices": Append(self.prices),
            "values": Append(self.values),
            "theoretical_values": Append(self._get_theoretical_values(self.A,self.l)),
            "supply": Append(self._get_supply(self.distributors, self.producers)),
            "supply_micro": Append(self._get_supply(self.distributors, self.producers, micro= True)),
            "accessible_supply": Append(self._get_supply(self.distributors)),
            "accessible_supply_micro": Append(self._get_supply(self.distributors, micro= True)),
            "avg_account": Append(np.average([dic["account"] for _, dic in self.persons.items()])),
            "min_account": Append(np.min([dic["account"] for _, dic in self.persons.items()])),
            "max_account": Append(np.max([dic["account"] for _, dic in self.persons.items()])),
            "avg_turnover_times": Append(self.turnover_times),
            "avg_endowments": Append(self._get_average_endowments(self.persons)),
            "A": self.A,
            "plans_in_progress": Append(np.zeros(self.settings["n_products"])),
            "goods_in_production": Append(np.zeros(self.settings["n_products"])),
            "actual_goods_in_production": Append(np.zeros(self.settings["n_products"])),
            "n_healthy": Append(self.settings["n_persons"]),
            "n_unhealthy": Append(0),
            "average_proficiencies": Append(self._get_average_abilities(self.persons)),
            "employment": Append(self.current_employment),
            "mean_consumption_frequencies": Append(self.consumption_frequencies),
            "mean_consumption_periods": Append(self.consumption_periods),
            "average_demand": Append(self._get_average(self.producers, self.distributors, key= "demand_signals")),
            "average_demand_producers": Append(self._get_average(self.producers, key= "demand_signals")),
            "average_demand_distributors": Append(self._get_average(self.distributors, key= "demand_signals")),
            "average_pending_inventories": Append(self._get_average(self.producers, self.distributors, key= "inventory")),
            "average_pending_inventories_distributors": Append(self._get_average(self.distributors, key= "inventory")),
            "average_pending_inventories_producers": Append(self._get_average(self.producers, key= "inventory")),
            "reorder_requests": Append(self.reorder_requests),
            "available_employment_by_sector": Append(self._get_available_employment_by_sector(self.producers)),
            "long_run_employment_by_sector": Append(self._get_available_employment_by_sector(self.producers)),
            "overall_busyness": Append(self.overall_busyness),
            "sectoral_busyness": Append(self._get_sectoral_busyness(self.producers)),
            "overall_weekly_busyness": Append(self.overall_weekly_busyness),
            "sectoral_weekly_busyness": Append(self._get_sectoral_busyness(self.producers, weekly= True)),
            "order_sizes": Append([np.average(orders) for orders in self.order_sizes]),
            "l": Append(self.l),
            "transfer_requests_by_sector": Append(self.transfer_requests_by_sector),
            "transfer_requests_by_sector_t": Replace(self.transfer_requests_by_sector_t),
            "eqb_employment": Append(self.eqb_employment),
            "min_hrly_output": Append(self.min_hrly_output),
            "busy_lower_bound": Append(self.busy_lower_bd),
            "busy_upper_bound": Append(self.busy_upper_bd),
            "long_run_activity": Append(self.long_run_sector_activity),
            "long_run_actual_activity": Append(self.long_run_actual_sector_activity),
            "t": Append(self.current_t)
        }

    # The stuff below this point are all just helper functions. 
    # Unless you're making your own or debugging something, you shouldn't ever have to look below here.

    def step(self) -> bool:
        """ 
        The basic step function. Retrieves json dictionaries for processing until the logged time step changes or the simulation finishes.
        """
        while True:
            if self.stdout_done and self.stderr_done:
                return True

            item = self.collector.get_next()

            if item.kind == "error":
                raise RuntimeError(f"{item.stream} reader failed: {item.payload}")

            if item.stream == "meta":
                if item.kind == "wait":
                    return False
                continue

            if item.stream == "stderr":
                if item.kind == "eof":
                    self.stderr_done = True
                else:
                    self.text_log.append((self.current_t, "stderr", item.payload))
                continue

            if item.stream == "stdout":
                if item.kind == "eof":
                    self.stdout_done = True
                else:
                    if self.is_logging:
                        with open(self.log_path, "a") as f:
                            print(item.payload, file= f)

                if item.kind == "json":
                    dic = item.payload
                    if self.current_t != dic["t"]:
                        if self.current_t == 0:
                            self.initialize_properties()
                            self.traj = self._declare_traj()
                        else:
                            self._update_hourly_stats()
                        self.current_t = dic["t"]
                        self.traj["t"] = Append(self.current_t)
                        self.t.append(self.current_t)
                        self._process_dic(dic)
                        return False
                    else:
                        self._process_dic(dic)
                else:
                    # item.kind == "text"
                    self.text_log.append((self.current_t, "stdout", item.payload))
                continue

            if self.stdout_done and self.stderr_done:
                return True



    def get_data(self):
        if hasattr(self, "traj"):
            return self.traj
        else:
            return {}


    def _get_args_from_settings(self):
        return [
            "-j",
            "-n", str(self.settings["n_time_steps"]),
            "-p", str(self.settings["n_persons"]),
            "-h", str(self.settings["init_working_day"]),
            "-w", str(self.settings["init_working_week"]),
            "-o", str(self.settings["n_commodities"]),
            "-m", str(self.settings["prods_per_machine"]),
            "-r", str(self.settings["n_producers"]),
            "-d", str(self.settings["n_distributors"]),
            "-s", str(self.settings["daily_sick_chance"]),
            "-v", str(self.settings["person_ability_stddev"])
        ]

    def _get_theoretical_values(self, A, l):
        n = A.shape[0]
        vals = inv(np.eye(n) - A.T)@l

        return vals

    def _get_supply(self, distributors, producers= None, micro= False):
        n_products = self.settings["n_products"]

        supply = np.zeros(n_products)
        key = "inventory"
        key += "_micro" if micro else ""

        for properties in distributors.values():
            inventory = properties[key]
            supply += inventory

        if producers != None:
            for properties in producers.values():
                inventory = properties[key]
                supply += inventory

        return supply

    def _get_average_endowments(self, persons):
        n_commodities = self.settings["n_commodities"]

        endowments = [dic["endowment"] for _,dic in persons.items()]
        itemwise_endowments = [[] for i in range(n_commodities)]
        for i in range(n_commodities):
            for j in range(len(persons)):
                itemwise_endowments[i].append(endowments[j][i])

        average_endowments = [np.average(itemwise_endowments[i]) for i in range(n_commodities)]
        return average_endowments

    def _get_average_abilities(self, persons):
        n_abilities = self.settings["n_abilities"]

        abilities = [dic["abilities"] for _,dic in persons.items()]

        abilitywise_profs = [[] for i in range(n_abilities)]
        for i in range(n_abilities):
            for j in range(len(persons)):
                abilitywise_profs[i].append(abilities[j][i])

        average_proficiencies = [np.average(abilitywise_profs[i]) for i in range(n_abilities)]
        return average_proficiencies

    def _get_available_employment_by_sector(self, producers):
        n_products = self.settings["n_products"]

        sectoral_employment = np.zeros(n_products)
        for _, properties in producers.items():
            cat = properties["catalog"]
            employees = properties["employees"]
            for prod_id in cat:
                sectoral_employment[prod_id] += employees

        return sectoral_employment

    def _get_sectoral_busyness(self, producers, weekly= False):
        n_products = self.settings["n_products"]
        key = "recent_weekly_busyness" if weekly else "recent_busyness"

        sectoral_busyness_data = [[] for i in range(n_products)]
        for _, properties in producers.items():
            cat = properties["catalog"]
            busyness = properties[key]
            for prod_id in cat:
                sectoral_busyness_data[prod_id].append(busyness)

        sectoral_busyness = np.array([np.average(sector) for sector in sectoral_busyness_data])
        return sectoral_busyness

    def _get_average(self, *firms, key= "demand_signals"):
        n_products = self.settings["n_products"]
        average_demands = np.zeros(n_products)
        for j in range(n_products):
            total_demand_j = []
            for firm_group in firms:
                total_demand_j += [firm_group[firm][key][j] for firm in firm_group if j in firm_group[firm]["catalog"]]
            average_demands[j] = np.average(total_demand_j)

        return average_demands

    # def _get_average(self, producers, distributors= None, key= "demand_signals", distributors_only= False):
    #     if distributors is None and distributors_only:
    #         raise Exception
    #     n_products = self.settings["n_products"]
    #     average_demands = np.zeros(n_products)
    #     for j in range(n_products):
    #         if distributors is not None:
    #             all_demands_distributor = [distributors[i][key][j] for i in distributors]
    #         if not distributors_only:
    #             all_demands_producer = [producers[i][key][j] for i in producers]
    #             if distributors is not None:
    #                 all_demands = all_demands_producer + all_demands_distributor
    #             else:
    #                 all_demands = all_demands_producer
    #         else:
    #             all_demands = all_demands_distributor
    #         average_demands[j] = np.average(all_demands)

    #     return average_demands

    def _update_data(self, key, val):
        if isinstance(val, np.ndarray) or isinstance(val, list):
            self.traj[key] = np.append(self.traj[key], [val], axis= 0)
        else:
            self.traj[key] = np.append(self.traj[key], val)

    def _get_dist_key(self, dist_id):
        return dist_id - self.settings["n_producers"]

    def _set_pending_inventories(self):
        for _, producer_dict in self.producers.items():
            producer_dict["pending_inventory"] = producer_dict["inventory"]+producer_dict["inc_inventory"]

        for _, distributor_dict in self.distributors.items():
            distributor_dict["pending_inventory"] = distributor_dict["inventory"]+distributor_dict["inc_inventory"]

    def get_expected_quantity(self, prod_id, quantity, n_workers):
        labor_hours_reqd = self.l[prod_id]*quantity
        work_hours_done = math.ceil(labor_hours_reqd / n_workers)
        quantity_produced = work_hours_done * n_workers / self.l[prod_id]
        return quantity_produced

