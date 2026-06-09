import logging
logger = logging.getLogger(__name__)
import numpy as np
from .Collector import Collector
from typing import Tuple
from scipy.linalg import inv, eig
from pathlib import Path
import math
import copy
from overseer.tools.dataclasses import Replace, Extend, Append

np.set_printoptions(
    precision=3,      # digits after decimal-ish
    suppress=True,    # avoid scientific notation when possible
    linewidth=200     # avoid wrapping rows too early
)

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
            "n_abilities": params.N_a,
            "productivity": params.productivity,
            "consump_epsilon": params.consump_epsilon,
            # the rest of these are d if pending_inventory_j else 0ependent variables
            "n_produced_goods": params.N_c,
            "n_machines": params.N_c // params.m_r,
            "n_products": 2 * params.N_c + params.N_c // params.m_r,
            "init_prices": params.init_prices
        }

        reasonable_logs = params.N_c < 10 and params.N_S <= 3000 and params.N_h < 100
        self.is_logging = params.is_logging and reasonable_logs and self.log_path is not None

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
            "endowment": np.zeros(self.settings["n_products"]),
            "abilities": np.zeros(self.settings["n_abilities"]),
            "health": "Healthy", # everyone starts in good health
            "recent_busyness": 0.0
        } for i in range(self.settings["n_persons"])}

        self.producers = {i: {
            "employees": 0,
            "inventory": np.zeros(self.settings["n_products"]),
            "pending_inventory": np.zeros(self.settings["n_products"]),
            "demand_signals": np.zeros(self.settings["n_products"]),
            "catalog": [],
            "recent_busyness": 0,
            "inc_inventory": np.zeros(self.settings["n_products"])
        } for i in range(self.settings["n_producers"])}

        self.distributors = {i: {
            "employees": 0,
            "inventory": np.zeros(self.settings["n_products"]),
            "pending_inventory": np.zeros(self.settings["n_products"]),
            "demand_signals": np.zeros(self.settings["n_products"]),
            "catalog": [],
            "recent_busyness": 0,
            "inc_inventory": np.zeros(self.settings["n_products"])
        } for i in range(self.settings["n_distributors"])}

        self.A = np.zeros((self.settings["n_products"], self.settings["n_products"]))
        self.l = np.zeros(self.settings["n_products"])
        self.b = np.zeros(self.settings["n_products"])
        self.consumption_frequencies = np.zeros(self.settings["n_products"])
        self.consumption_periods = np.zeros(self.settings["n_products"])
        self.order_sizes = [[] for i in range(self.settings["n_products"])]
        self.transfer_requests_by_sector = np.zeros(self.settings["n_products"])
        self.transfer_requests_by_sector_t = np.array([])
        self.active_plans = {i: {"plans": 0, "quantity": 0} for i in range(self.settings["n_products"])}
        self.reorder_requests = np.zeros(self.settings["n_products"])
        self.overall_busyness = 0
        self.overall_busyness_data = []
        self.overall_weekly_busyness = 0
        self.long_run_employment_by_sector = np.zeros(self.settings["n_produced_goods"]+self.settings["n_machines"]+1)
        self.long_run_sector_activity = np.zeros(self.settings["n_products"])

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
                    coords_str = dic["coords"]
                    coords = tuple(int(num) for num in coords_str[1:len(coords_str)-1].split(","))
                    # coords = tuple(dic.get("coords"))
                    i, j = coords
                    a_ij = dic["value"]
                    self.A[i][j] = a_ij

                if label == "l":
                    i = dic["prod_id"]
                    l_i = dic["value"]

                    self.l[i] = l_i

                if label == "b":
                    i = dic["prod_id"]
                    b_i = dic["value"]
                    self.b[i] = b_i

                if label == "price":
                    id = dic["product_id"]
                    val = dic["price_per_unit"]
                    self.prices[id] = val

                if label == "new_price":
                    prod_id = dic["product_id"]
                    price = dic["price"]
                    self.prices[prod_id] = price

                if label == "mean_consumption_frequency":
                    id = dic["product_id"]
                    val = dic["value"]
                    self.consumption_frequencies[id] = val
                    self.consumption_periods[id] = 1/max(val, 1e-5)

                if label == "employment":
                    self.current_employment = dic["total"]

            case "Person":
                if label == "age":
                    self.persons[id]["health"] = values[0]

                if label == "account":
                    self.persons[id]["account"] = dic["value"]

                if label == "health_status":
                    self.persons[id]["health"] = dic["status"]

                if label == "consumption":
                    prod_id = dic["product_id"]
                    amt = dic["quantity"]
                    self.persons[id]["endowment"][prod_id] = amt

                if label == "ability":
                    ability_id = dic["ability"]
                    val = dic["value"]
                    person_id = dic['id']
                    self.persons[person_id]["abilities"][ability_id] = val

                if label == "inventory":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    person_id = dic['id']
                    self.persons[person_id]["endowment"][prod_id] = amt

                if label == "purchase":
                    prod_id = dic["product_id"]
                    amt = dic["quantity"]
                    self.persons[id]["endowment"][prod_id] += amt
                    cost = self.prices[prod_id]*amt
                    self.persons[id]["account"] -= cost

                if label == "hours_worked":
                    self.persons[id]["account"] += dic["hours"]

            case "Producer":
                if label == "inventory_level":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    producer_id = dic['id']
                    self.producers[producer_id]["inventory"][prod_id] = amt

                if label == "inventory_reduction":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    # distributor_inventories[id-n_producers][prod_id] -= amt

                if label == "catalog_addition":
                    product_id = dic["product_id"]
                    self.producers[id]["catalog"].append(product_id)

                if label == "pursued_plan":
                    producer_id = id
                    customer_id = dic["customer_id"]
                    is_distributor = (customer_id >= self.settings["n_producers"])
                    if is_distributor:
                        customer_id = self._get_dist_key(customer_id)
                    prod_id = dic["product_id"]
                    amt = dic["quantity"]
                    n_workers = dic["num_workers"]

                    self.active_plans[prod_id]["plans"] += 1
                    self.active_plans[prod_id]["quantity"] += amt
                    self.long_run_sector_activity[prod_id] += amt

                    if is_distributor:
                        self.distributors[customer_id]["inc_inventory"][prod_id] += amt
                    else:
                        self.producers[customer_id]["inc_inventory"][prod_id] += amt

                if label == "ended_plan":
                    prod_id = dic["product_id"]
                    amt = dic["quantity"]
                    self.active_plans[prod_id]["plans"] -= 1
                    self.active_plans[prod_id]["quantity"] -= amt

                if label == "shipment_received":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    self.producers[id]["inc_inventory"][prod_id] -= amt

                if label == "current_demand":
                    prod_id = dic["product_id"]
                    demand = dic["demand"]
                    self.producers[id]["demand_signals"][prod_id] = demand

                if label in {"reorder", "reorder_failure"}:
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    self.reorder_requests[prod_id] += 1

                if label == "newly_employed":
                    self.producers[id]["employees"] += 1

                if label == "busyness":
                    firm_busyness = dic["firm_busyness"]
                    overall_busyness = dic["societal_busyness"]
                    transfers_available = dic["max_workers_for_transfer"]
                    self.producers[id]["recent_busyness"] = firm_busyness
                    self.overall_busyness_data.append(firm_busyness)
                    self.overall_busyness = overall_busyness

                if label == "accepted_order":
                    prod_id = dic["product_id"]
                    time = dic["offered_turnaround_time"]
                    self.turnover_times[prod_id].append(time)

                if label == "transfer_request":
                    cat = self.producers[id]["catalog"]
                    for i in cat:
                        self.transfer_requests_by_sector[i] +=  1
                        if len(self.transfer_requests_by_sector_t) == 0 or self.current_t != self.transfer_requests_by_sector_t[-1]:
                            self.transfer_requests_by_sector_t = np.append(self.transfer_requests_by_sector_t, self.current_t)

                if label == "transfer":
                    old_emp = dic["old_workplace_id"]
                    old_emp_is_distributor = (old_emp >= self.settings["n_producers"])
                    if old_emp_is_distributor:
                        old_emp = self._get_dist_key(old_emp)
                        self.distributors[old_emp]["employees"] -= 1
                    else:
                        self.producers[old_emp]["employees"] -= 1

                    new_emp = dic["new_workplace_id"]
                    new_emp_is_distributor = (new_emp >= self.settings["n_producers"])
                    if new_emp_is_distributor:
                        new_emp = self._get_dist_key(new_emp)
                        self.distributors[new_emp]["employees"] += 1
                    else:
                        self.producers[new_emp]["employees"] += 1

            case "Distributor":
                if label == "inventory_level":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    distributor_id = dic['id']
                    dist_key = self._get_dist_key(distributor_id)
                    self.distributors[dist_key]["inventory"][prod_id] = amt

                if label == "inventory_reduction":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    dist_key = self._get_dist_key(id)
                    self.distributors[dist_key]["inventory"][prod_id] -= amt

                if label == "catalog":
                    product_ids_str = dic["product_ids"]
                    product_ids_str_list = product_ids_str.split(",")
                    product_ids = [int(product_id) for product_id in product_ids_str_list]
                    dist_key = self._get_dist_key(id)
                    self.distributors[dist_key]["catalog"] = product_ids

                if label == "accepted_order":
                    prod_id = dic["product_id"]
                    time = dic["offered_turnaround_time"]
                    self.turnover_times[prod_id].append(time)

                if label == "shipment_received":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["inc_inventory"][prod_id] -= amt

                if label == "catalog_addition":
                    prod_id = dic["product_id"]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["catalog"].append(prod_id)

                if label == "current_demand":
                    prod_id = dic["product_id"]
                    demand = dic["demand"]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["demand_signals"][prod_id] = demand

                if label in {"reorder", "reorder_failure"}:
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    self.reorder_requests[prod_id] += 1

                if label == "newly_employed":
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["employees"] += 1

                if label == "busyness":
                    firm_busyness = dic["firm_busyness"]
                    overall_busyness = dic["societal_busyness"]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["recent_busyness"] = firm_busyness
                    self.overall_busyness_data.append(firm_busyness)
                    self.overall_busyness = overall_busyness

                if label == "transfer":
                    old_emp = dic["old_workplace_id"]
                    old_emp_is_distributor = (old_emp >= self.settings["n_producers"])
                    if old_emp_is_distributor:
                        old_emp = self._get_dist_key(old_emp)
                        self.distributors[old_emp]["employees"] -= 1
                    else:
                        self.producers[old_emp]["employees"] -= 1

                    new_emp = dic["new_workplace_id"]
                    new_emp_is_distributor = (new_emp >= self.settings["n_producers"])
                    if new_emp_is_distributor:
                        new_emp = self._get_dist_key(new_emp)
                        self.distributors[new_emp]["employees"] += 1
                    else:
                        self.producers[new_emp]["employees"] += 1

                if label == "transfer_request":
                    dist_id = self._get_dist_key(id)
                    cat = self.distributors[dist_id]["catalog"]
                    for i in cat:
                        self.transfer_requests_by_sector[i] +=  1
                        if len(self.transfer_requests_by_sector_t) == 0 or self.current_t != self.transfer_requests_by_sector_t[-1]:
                            self.transfer_requests_by_sector_t = np.append(self.transfer_requests_by_sector_t, self.current_t)

                if label == "pursued_plan":
                    customer_id = dic["customer_id"]
                    is_distributor = (customer_id >= self.settings["n_producers"])
                    if is_distributor:
                        customer_id = self._get_dist_key(customer_id)
                    prod_id = dic["product_id"]
                    amt = dic["quantity"]

                    self.active_plans[prod_id]["plans"] += 1
                    self.active_plans[prod_id]["quantity"] += amt
                    self.long_run_sector_activity[prod_id] += amt

                    if is_distributor:
                        self.distributors[customer_id]["inc_inventory"][prod_id] += amt
                    else:
                        self.producers[customer_id]["inc_inventory"][prod_id] += amt

                if label == "ended_plan":
                    prod_id = dic["product_id"]
                    amt = dic["quantity"]
                    self.active_plans[prod_id]["plans"] -= 1
                    self.active_plans[prod_id]["quantity"] -= amt


    def _update_hourly_stats(self):
        """ 
        Updates the trajectories dictionary.
        """
        producer_supply = self._get_producer_supply()
        producer_supply_machines = self._get_producer_supply(machines= True)

        consumer_goods_supply = self._get_distributor_supply()
        distributor_unshelved_supply = self._get_distributor_supply(produced= True)

        average_demands = self._get_overall_demand()
        average_demands_producers = self._get_producer_demands()
        average_machine_demand = self._get_producer_demands(machines= True)
        average_demands_distributors = self._get_distributor_demands(produced= True)

        self._set_pending_inventories()
        average_pending_inventories_all = self._get_overall_pending_inventory()
        average_pending_inventories_distributors = self._get_distributor_pending_inventories(produced= True)
        average_pending_inventories_producers = self._get_producer_pending_inventories()
        average_pending_inventories_consumption = self._get_distributor_pending_inventories()

        accounts = [dic["account"] for _, dic in self.persons.items()]
        health_statuses = [0 if dic["health"] == "Healthy" else 1 for _, dic in self.persons.items()]
        n_unhealthy = sum(health_statuses)
        n_healthy = len(self.persons) - n_unhealthy

        average_endowments = self._get_average_endowments(self.persons)
        average_proficiencies = self._get_average_abilities(self.persons)

        plans_in_motion = [self.active_plans[i]["plans"] for i in range(self.settings["n_products"])]
        quantities_in_prod = [self.active_plans[i]["quantity"] for i in range(self.settings["n_products"])]

        sectoral_employment = self._get_available_employment_by_sector()
        sectoral_busyness = self._get_sectoral_busyness()

        order_size_averages = np.array([np.average(orders) for orders in self.order_sizes])

        busyness_data = np.asarray(self.overall_busyness_data)
        if len(self.overall_busyness_data) > 0:
            low, hi = np.quantile(busyness_data, [0.005, 0.995])
            overall_busyness_bins = np.linspace(low, hi, 100)
        else:
            overall_busyness_bins = np.array([0.5])

        self.long_run_employment_by_sector += sectoral_employment

        n_prod_goods = self.settings["n_produced_goods"]

        self.traj = {
            "producer_goods_prices": Append(self.prices[:n_prod_goods]),
            "consumption_goods_prices": Append(self.prices[n_prod_goods:2*n_prod_goods]),
            "machine_prices": Append(self.prices[2*n_prod_goods:]),

            "producer_goods_values": Append(self.values[:n_prod_goods]),
            "consumption_goods_values": Append(self.values[n_prod_goods:2*n_prod_goods]),
            "machine_values": Append(self.values[2*n_prod_goods:]),

            "b": Append(self.b),
            "producer_supply": Append(producer_supply),
            "producer_supply_machines": Append(producer_supply_machines),
            "consumer_goods_supply": Append(consumer_goods_supply),
            "distributor_unshelved_supply": Append(distributor_unshelved_supply),

            "avg_account": Append(np.average(accounts)),
            "min_account": Append(np.min(accounts)),
            "max_account": Append(np.max(accounts)),
            "avg_endowments": Append(average_endowments),
            "plans_in_progress": Append(plans_in_motion),
            "goods_in_production": Append(quantities_in_prod),
            "n_healthy": Append(n_healthy),
            "n_unhealthy": Append(n_unhealthy),
            "average_proficiencies": Append(average_proficiencies),
            "employment": Append(self.current_employment),
            "mean_consumption_frequencies": Append(self.consumption_frequencies),
            "mean_consumption_periods": Append(self.consumption_periods),

            "average_demand": Append(average_demands),
            "average_machine_demand": Append(average_machine_demand),
            "average_demand_producers": Append(average_demands_producers),
            "average_demand_distributors": Append(average_demands_distributors),

            "average_pending_inventories": Append(average_pending_inventories_all),
            "average_pending_inventories_distributors": Append(average_pending_inventories_distributors),
            "average_pending_inventories_producers": Append(average_pending_inventories_producers),
            "average_pending_inventories_consumption": Append(average_pending_inventories_consumption),

            "reorder_requests": Append(self.reorder_requests),
            "available_employment_by_sector": Append(sectoral_employment),
            "sectoral_busyness": Append(sectoral_busyness),
            "overall_busyness": Append(self.overall_busyness),
            "busyness_data": Replace(self.overall_busyness_data),
            "overall_busyness_bins": Replace(overall_busyness_bins),
            "order_sizes": Append(order_size_averages),
            "l": Append(self.l),
            "transfer_requests_by_sector": Append(self.transfer_requests_by_sector),
            "long_run_employment_by_sector": Append(self.long_run_employment_by_sector / max(self.current_t, 1)),
            "eqb_employment": Append(self.eqb_employment),
            "min_hrly_output": Append(self.min_hrly_output),
            "busy_lower_bound": Append(self.busy_lower_bd),
            "busy_upper_bound": Append(self.busy_upper_bd),
            "long_run_activity": Append(self.long_run_sector_activity / max(self.current_t, 1)),
            "transfer_requests_by_sector_t": Replace(self.transfer_requests_by_sector_t),
            "A": Replace(self.A)
        }

        self.transfer_requests_by_sector = np.zeros(self.settings["n_products"])
        self.reorder_requests = np.zeros(self.settings["n_products"])

    def initialize_properties(self):
        N = self.settings["n_persons"]
        logger.info(f"{self.consumption_frequencies=}")
        net_weekly_demand = N*24*7*self.consumption_frequencies
        gross_weekly_demand = inv(np.eye(self.settings["n_products"]) - self.A)@net_weekly_demand
        sectoral_weekly_labor_req_raw = self.l * gross_weekly_demand
        prod_goods_sectoral_weekly_labor_req = list(sectoral_weekly_labor_req_raw[0:self.settings["n_produced_goods"]])
        machine_sectoral_weekly_labor_req = list(sectoral_weekly_labor_req_raw[2*self.settings["n_produced_goods"]:])

        overall_sectoral_weekly_labor_req = []

        lo = self.settings["n_produced_goods"]
        hi = 2*self.settings["n_produced_goods"]
        overall_sectoral_weekly_labor_req.extend(prod_goods_sectoral_weekly_labor_req)
        overall_sectoral_weekly_labor_req.append(sum(sectoral_weekly_labor_req_raw[lo:hi]))
        overall_sectoral_weekly_labor_req.extend(machine_sectoral_weekly_labor_req)
        overall_sectoral_weekly_labor_req = np.asarray(overall_sectoral_weekly_labor_req)
        min_hrly_output = gross_weekly_demand / (24*7)

        self.eqb_employment = overall_sectoral_weekly_labor_req / (8*5)
        self.busy_lower_bd = self.settings["consump_epsilon"]*(8*5 / (24*7))
        self.busy_upper_bd = (8*5 / (24*7))
        self.min_hrly_output = min_hrly_output
        dim = self.A.shape[0]
        self.values = inv(np.eye(dim) - self.A.T)@self.l
        logger.info(f"A =\n {self.A}, l=\n{self.l}")
        logger.info(f"values = {self.values=}")

        (evals, evecs) = eig(self.A)
        idx = np.argmax(evals.real)
        r_hat = np.real(evals[idx])
        logger.info(f"Rho(A) = {r_hat}")

        if self.settings["init_prices"] == "values":
            self.b = self.consumption_frequencies

        if self.settings["init_prices"] == "equilibrium_prices":
            M = self.A + np.linalg.outer(self.b, self.l)
            (evals, evecs) = eig(M.T)
            idx = np.argmax(evals.real)
            r_hat = np.real(evals[idx])
            epr = 1/r_hat - 1
            # logger.info(f"EPR = {epr}, b = {self.b}, A = {self.A}, l = {self.l}, p = {self.prices}")

        for dist_dict in self.distributors.values():
            logger.info(f"{dist_dict["catalog"]=}")


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
                            self.traj = {
                                "A": Replace(self.A)
                            }
                            self._update_hourly_stats()
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
            "-g", str(self.settings["n_commodities"]),
            "-m", str(self.settings["prods_per_machine"]),
            "-r", str(self.settings["n_producers"]),
            "-d", str(self.settings["n_distributors"]),
            "-s", str(self.settings["daily_sick_chance"]),
            "-a", str(self.settings["n_abilities"]),
            "-v", str(self.settings["person_ability_stddev"]),
            "--production_difficulty", str(self.settings["productivity"]),
            "--consumption_demand", str(self.settings["consump_epsilon"]),
            "--init_prices", str(self.settings["init_prices"])
        ]

    def _get_theoretical_values(self, A, l):
        n = A.shape[0]
        vals = inv(np.eye(n) - A.T)@l

        return vals

    def _get_distributor_supply(self, produced= False):
        n_prod_goods = self.settings["n_produced_goods"]
        if produced:
            idx_low = 0
            idx_high = n_prod_goods
        else:
            idx_low = n_prod_goods
            idx_high = 2*n_prod_goods

        supply = np.zeros(n_prod_goods)
        for properties in self.distributors.values():
            inventory = properties["inventory"]
            supply += inventory[idx_low:idx_high]

        return supply

    def _get_producer_supply(self, machines= False):
        n_prod_goods = self.settings["n_produced_goods"]
        n_machines = self.settings["n_machines"]
        if machines:
            idx_low = 2*n_prod_goods
            idx_high = idx_low + n_machines
        else:
            idx_low = 0
            idx_high = n_prod_goods

        supply = np.zeros(n_prod_goods) if not machines else np.zeros(n_machines)
        for properties in self.producers.values():
            inventory = properties["inventory"]
            try:
                supply += inventory[idx_low:idx_high]
            except ValueError:
                logger.info(f"{inventory=}")
                logger.info(f"{supply=}")
                logger.info(f"{idx_low=}")
                logger.info(f"{idx_high=}")
                logger.info(f"{machines=}")
                raise ValueError("Same bug")

        return supply

    def _get_producer_demands(self, machines= False):
        n_prod_goods = self.settings["n_produced_goods"]
        if machines:
            idx_low = 2*n_prod_goods
            idx_high = self.settings["n_products"]
        else:
            idx_low = 0
            idx_high = n_prod_goods

        average_demands = np.zeros(self.settings["n_machines"]) if machines else np.zeros(self.settings["n_produced_goods"])
        idx_list = list(range(idx_low, idx_high))
        for i, j in enumerate(idx_list):
            total_demand_j = [producer_dict["demand_signals"][j] for producer_dict in self.producers.values()]# if j in producer_dict["catalog"]]
            average_demands[i] = np.average(total_demand_j) if total_demand_j else 0

        return average_demands

    def _get_producer_pending_inventories(self, machines= False):
        n_prod_goods = self.settings["n_produced_goods"]
        if machines:
            idx_low = 2*n_prod_goods
            idx_high = self.settings["n_products"]
        else:
            idx_low = 0
            idx_high = n_prod_goods

        average_pending_inventories = np.zeros(self.settings["n_machines"]) if machines else np.zeros(self.settings["n_produced_goods"])
        idx_list = list(range(idx_low, idx_high))
        for i, j in enumerate(idx_list):
            pending_inventory_j = [producer_dict["pending_inventory"][j] for producer_dict in self.producers.values()]# if j in producer_dict["catalog"]]
            average_pending_inventories[i] = np.average(pending_inventory_j) if pending_inventory_j else 0

        return average_pending_inventories

    def _get_distributor_demands(self, produced= False):
        n_prod_goods = self.settings["n_produced_goods"]
        if produced:
            idx_low = 0
            idx_high = n_prod_goods
        else:
            idx_low = n_prod_goods
            idx_high = 2*n_prod_goods

        average_demands = np.zeros(self.settings["n_produced_goods"])
        idx_list = list(range(idx_low, idx_high))
        for i,j in enumerate(idx_list):
            total_demand_j = [dist_dict["demand_signals"][j] for dist_dict in self.distributors.values()]# if j in dist_dict["catalog"]]
            average_demands[i] = np.average(total_demand_j) if total_demand_j else 0

        return average_demands

    def _get_distributor_pending_inventories(self, produced= False):
        n_prod_goods = self.settings["n_produced_goods"]
        if produced:
            idx_low = 0
            idx_high = n_prod_goods
        else:
            idx_low = n_prod_goods
            idx_high = 2*n_prod_goods

        average_pending_inventories = np.zeros(self.settings["n_produced_goods"])
        idx_list = list(range(idx_low, idx_high))
        for i,j in enumerate(idx_list):
            pending_inventory_j = [dist_dict["pending_inventory"][j] for dist_dict in self.distributors.values()]# if j in dist_dict["catalog"]]
            # pending_inventory_j2 = [dist_dict["pending_inventory"][j] for dist_dict in self.distributors.values()]
            average_pending_inventories[i] = np.average(pending_inventory_j) if pending_inventory_j else 0

        return average_pending_inventories

    def _get_overall_demand(self):
        average_demands_producers = self._get_producer_demands(machines= False)
        average_demands_distributors = self._get_distributor_demands(produced= True)
        return average_demands_producers + average_demands_distributors

    def _get_overall_pending_inventory(self):
        average_pending_inventory_producers = self._get_producer_pending_inventories(machines= False)
        average_pending_inventories_distributors = self._get_distributor_pending_inventories(produced= True)
        return average_pending_inventory_producers + average_pending_inventories_distributors

    def _get_average(self, *firms, key= "demand_signals"):
        n_products = self.settings["n_products"]
        average_demands = np.zeros(n_products)
        for j in range(n_products):
            total_demand_j = []
            for firm_group in firms:
                total_demand_j += [firm_group[firm][key][j] for firm in firm_group if j in firm_group[firm]["catalog"]]
            average_demands[j] = np.average(total_demand_j)

        return average_demands


    def _get_supply(self, distributors, producers= None):
        n_products = self.settings["n_products"]

        supply = np.zeros(n_products)

        for properties in distributors.values():
            inventory = properties["inventory"]
            supply += inventory

        if producers != None:
            for properties in producers.values():
                inventory = properties["inventory"]
                supply += inventory

        return supply

    def _get_average_endowments(self, persons):

        n_prod_goods = self.settings["n_produced_goods"]
        idx_low = n_prod_goods
        idx_high = 2*n_prod_goods
        consumer_goods_idxs = list(range(idx_low, idx_high))

        n_commodities = self.settings["n_commodities"]

        endowments = [dic["endowment"] for _,dic in persons.items()]
        itemwise_endowments = [[] for i in range(n_prod_goods)]
        for i, idx in enumerate(consumer_goods_idxs):
            for j in range(len(persons)):
                itemwise_endowments[i].append(endowments[j][idx])

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

    def _get_available_employment_by_sector(self):
        n_sectors = self.settings["n_produced_goods"]+self.settings["n_machines"]+1

        sectoral_employment = np.zeros(n_sectors)
        for properties in self.producers.values():
            cat = properties["catalog"]
            employees = properties["employees"]
            for prod_id in cat:
                if prod_id >= 2*self.settings["n_produced_goods"]:
                    machine_id = prod_id - self.settings["n_produced_goods"]
                    sectoral_employment[machine_id] += employees
                else:
                    sectoral_employment[prod_id] += employees

        for properties in self.distributors.values():
            sectoral_employment[n_sectors-1] += properties["employees"]

        return sectoral_employment

    def _get_sectoral_busyness(self):
        n_sectors = self.settings["n_produced_goods"]+self.settings["n_machines"]+1

        sectoral_busyness_data = [[] for _ in range(n_sectors)]
        for properties in self.producers.values():
            cat = properties["catalog"]
            busyness = properties["recent_busyness"]
            for prod_id in cat:
                if prod_id >= 2*self.settings["n_produced_goods"]:
                    machine_id = prod_id - self.settings["n_produced_goods"]
                    sectoral_busyness_data[machine_id].append(busyness)
                else:
                    sectoral_busyness_data[prod_id].append(busyness)

        for properties in self.distributors.values():
            sectoral_busyness_data[n_sectors-1].append(properties["recent_busyness"])

        sectoral_busyness = np.array([np.average(sector) for sector in sectoral_busyness_data])
        return sectoral_busyness


    def _set_pending_inventories(self):
        for _, producer_dict in self.producers.items():
            producer_dict["pending_inventory"] = producer_dict["inventory"]+producer_dict["inc_inventory"]

        for _, distributor_dict in self.distributors.items():
            distributor_dict["pending_inventory"] = distributor_dict["inventory"]+distributor_dict["inc_inventory"]

    def _get_good_type_and_idx(self, id):
        n_produced_goods = self.settings["n_produced_goods"]
        if id >= 2*n_produced_goods:
            # machine
            return "machine", id-2*n_produced_goods
        elif id >= n_produced_goods:
            # consumer good
            return "consumer_good", id-n_produced_goods
        else:
            return "production_good", id

    def _get_dist_key(self, dist_id):
        return dist_id - self.settings["n_producers"]

