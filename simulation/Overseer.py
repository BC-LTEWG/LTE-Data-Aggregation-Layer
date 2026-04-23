import numpy as np
from .Collector import Collector
from typing import Tuple
from scipy.linalg import inv

class Overseer:
    """ The guy who watches the LTE and keeps track of the data. """
    def __init__(self, bin_path, params, log_path: str | None = None):
        self.params = params
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
            "n_products": params.N_c + params.N_c // params.m_r,
        }

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
        } for i in range(self.settings["n_producers"])}

        self.distributors = {i: {
            "employees": 0,
            "inventory": np.zeros(self.settings["n_products"]),
            "inventory_micro": np.zeros(self.settings["n_products"]),
            "pending_inventory": np.zeros(self.settings["n_products"]),
            "demand_signals": np.zeros(self.settings["n_products"]),
            "catalog": [],
            "recent_busyness": 0,
        } for i in range(self.settings["n_distributors"])}


        self.A = np.zeros((self.settings["n_products"], self.settings["n_products"]))
        self.l = np.zeros(self.settings["n_products"])
        self.consumption_frequencies = np.zeros(self.settings["n_products"])
        self.consumption_periods = np.zeros(self.settings["n_products"])
        self.order_sizes = [[] for i in range(self.settings["n_products"])]
        self.transfer_requests_by_sector = np.zeros(self.settings["n_products"])
        self.transfer_requests_by_sector_t = np.array([])
        self.active_plans = {i: {"plans": 0, "quantity": 0} for i in range(self.settings["n_products"])}
        self.reorder_requests = np.zeros(self.settings["n_products"])
        self.overall_busyness = 0


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
                    self.persons[id]["endowment"][prod_id] -= amt

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
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]

                    self.active_plans[prod_id]["plans"] += 1
                    self.active_plans[prod_id]["quantity"] += amt

                if label == "ended_plan":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    amt = pair[1]
                    self.active_plans[prod_id]["plans"] -= 1
                    self.active_plans[prod_id]["quantity"] -= amt

                if label == "shipment_received":
                    prod_id = dic["product_id"]
                    amt = dic["amount"]
                    self.producers[id]["inventory_micro"][prod_id] += amt

                if label == "current_demand":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    demand = pair[1]
                    self.producers[id]["demand_signals"][prod_id] = demand

                if label == "pending_inventory":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    threshold = pair[1]
                    self.producers[id]["pending_inventory"][prod_id] = threshold

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
                    self.distributors[dist_id]["inventory_micro"][prod_id] += amt

                if label == "current_demand":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    demand = pair[1]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["demand_signals"][prod_id] = demand

                if label == "pending_inventory":
                    pair = list(dic.items())[-1]
                    prod_str = pair[0]
                    prod_id = int(prod_str.split('_')[1])
                    threshold = pair[1]
                    dist_id = self._get_dist_key(id)
                    self.distributors[dist_id]["pending_inventory"][prod_id] = threshold

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
        average_pending_inventories = self._get_average(self.producers, self.distributors, key= "pending_inventory")

        accounts = [dic["account"] for _, dic in self.persons.items()]
        health_statuses = [0 if dic["health"] == "Healthy" else 1 for _, dic in self.persons.items()]
        n_unhealthy = sum(health_statuses)
        n_healthy = len(self.persons) - n_unhealthy

        average_endowments = self._get_average_endowments(self.persons)
        average_proficiencies = self._get_average_abilities(self.persons)

        plans_in_motion = [self.active_plans[i]["plans"] for i in range(self.settings["n_products"])]
        quantities_in_prod = [self.active_plans[i]["quantity"] for i in range(self.settings["n_products"])]

        sectoral_employment = self._get_available_employment_by_sector(self.producers)
        sectoral_busyness = self._get_sectoral_busyness(self.producers)

        order_size_averages = np.array([np.average(orders) for orders in self.order_sizes])

        self._update_data("prices", self.prices)
        self._update_data("values", self.traj["values"][-1])
        self._update_data("supply", overall_supply)
        self._update_data("supply_micro", overall_supply_micro)
        self._update_data("accessible_supply", accessible_supply)
        self._update_data("accessible_supply_micro", accessible_supply_micro)
        self._update_data("avg_account", np.average(accounts))
        self._update_data("avg_endowments", average_endowments)
        self._update_data("plans_in_progress", plans_in_motion)
        self._update_data("goods_in_production", quantities_in_prod)
        self._update_data("n_healthy", n_healthy)
        self._update_data("n_unhealthy", n_unhealthy)
        self._update_data("average_proficiencies", average_proficiencies)
        self._update_data("employment", self.current_employment)
        self._update_data("mean_consumption_frequencies", self.consumption_frequencies)
        self._update_data("mean_consumption_periods", self.consumption_periods)
        self._update_data("average_demand", average_demands)
        self._update_data("average_pending_inventories", average_pending_inventories)
        self._update_data("reorder_requests", self.reorder_requests)
        self._update_data("available_employment_by_sector", sectoral_employment)
        self._update_data("sectoral_busyness", sectoral_busyness)
        self._update_data("overall_busyness", self.overall_busyness)
        self._update_data("order_sizes", order_size_averages)
        self._update_data("l", self.traj["l"][-1])
        self._update_data("transfer_requests_by_sector", self.transfer_requests_by_sector)
        self.traj["transfer_requests_by_sector_t"] = self.transfer_requests_by_sector_t

    def _declare_traj(self):
        """ 
        After time = 0 finishes, the trajectories dictionary is initialized from this function. You would need to add new ones to this if making your own.
        """
        return {
            "prices": np.array([self.prices]),
            "values": np.array([self.prices]),
            "theoretical_values": np.array([self._get_theoretical_values(self.A,self.l)]),
            "supply": np.array([self._get_supply(self.distributors, self.producers)]),
            "supply_micro": np.array([self._get_supply(self.distributors, self.producers, micro= True)]),
            "accessible_supply": np.array([self._get_supply(self.distributors)]),
            "accessible_supply_micro": np.array([self._get_supply(self.distributors, micro= True)]),
            "avg_account": np.array([np.average([dic["account"] for _, dic in self.persons.items()])]),
            "avg_turnover_times": np.array([self.turnover_times]),
            "avg_endowments": np.array([self._get_average_endowments(self.persons)]),
            "A": self.A,
            "plans_in_progress": np.array([np.zeros(self.settings["n_products"])]),
            "goods_in_production": np.array([np.zeros(self.settings["n_products"])]),
            "n_healthy": np.array([self.settings["n_persons"]]),
            "n_unhealthy": np.array([0]),
            "average_proficiencies": np.array([self._get_average_abilities(self.persons)]),
            "employment": np.array([self.current_employment]),
            "mean_consumption_frequencies": np.array([self.consumption_frequencies]),
            "mean_consumption_periods": np.array([self.consumption_periods]),
            "average_demand": np.array([np.zeros(self.settings["n_products"])]),
            "average_pending_inventories": np.array([np.zeros(self.settings["n_products"])]),
            "reorder_requests": np.array([self.reorder_requests]),
            "available_employment_by_sector": np.array([self._get_available_employment_by_sector(self.producers)]),
            "overall_busyness": np.array([self.overall_busyness]),
            "sectoral_busyness": np.array([self._get_sectoral_busyness(self.producers)]),
            "order_sizes": np.array([[np.average(orders) for orders in self.order_sizes]]),
            "l": np.array([self.l]),
            "transfer_requests_by_sector": np.array([self.transfer_requests_by_sector]),
            "transfer_requests_by_sector_t": self.transfer_requests_by_sector_t
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
                elif item.kind == "json":
                    dic = item.payload
                    if self.current_t != dic["t"]:
                        if self.current_t == 0:
                            self.traj = self._declare_traj()
                        else:
                            self._update_hourly_stats()
                        self.current_t = dic["t"]
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
            return self.traj, self.t
        else:
            return {}, self.t


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

    def _get_sectoral_busyness(self, producers):
        n_products = self.settings["n_products"]

        sectoral_busyness_data = [[] for i in range(n_products)]
        for _, properties in producers.items():
            cat = properties["catalog"]
            busyness = properties["recent_busyness"]
            for prod_id in cat:
                sectoral_busyness_data[prod_id].append(busyness)

        sectoral_busyness = np.array([np.average(sector) for sector in sectoral_busyness_data])
        return sectoral_busyness

    def _get_average(self, producers, distributors, key= "demand_signals"):
        n_products = self.settings["n_products"]
        average_demands = np.zeros(n_products)
        for j in range(n_products):
            all_demands_producer = [producers[i][key][j] for i in producers]
            all_demands_distributor = [distributors[i][key][j] for i in distributors]
            all_demands = all_demands_producer + all_demands_distributor
            average_demands[j] = np.average(all_demands)

        return average_demands

    def _update_data(self, key, val):
        if isinstance(val, np.ndarray) or isinstance(val, list):
            self.traj[key] = np.append(self.traj[key], [val], axis= 0)
        else:
            self.traj[key] = np.append(self.traj[key], val)

    def _get_dist_key(self, dist_id):
        return dist_id - self.settings["n_producers"]
