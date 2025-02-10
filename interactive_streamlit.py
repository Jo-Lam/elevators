from itertools import product

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# -----------------------------------------
# Simulation logic
# -----------------------------------------

def travel_time(floor_a, floor_b, elevator_speed):
    """Compute travel time between two floors given elevator speed (floors/min)."""
    return abs(floor_a - floor_b) / elevator_speed

class Elevator:
    def __init__(self, elevator_id, start_floor):
        self.elevator_id = elevator_id
        self.current_floor = start_floor
        self.time_next_available = 0.0

    def assign_call(self, call_time, origin, destination, elevator_speed, door_time):
        """
        Assign a call to this elevator, return the waiting time.
        - call_time: when the user pressed the button
        - self.time_next_available: earliest the elevator can start moving again
        """
        start_move_time = max(call_time, self.time_next_available)

        # Travel from current floor to origin
        time_to_origin = travel_time(self.current_floor, origin, elevator_speed)
        arrival_to_origin_time = start_move_time + time_to_origin

        # Passenger waiting time
        waiting_time = arrival_to_origin_time - call_time

        # Travel from origin to destination
        time_to_destination = travel_time(origin, destination, elevator_speed)
        arrival_to_destination_time = arrival_to_origin_time + time_to_destination + door_time

        # Update elevator state
        self.current_floor = destination
        self.time_next_available = arrival_to_destination_time

        return waiting_time

def default_get_destination(origin, floors):
    """
    Destination chosen uniformly from floors != origin.
    Ensures we never start and end on the same floor.
    """
    possible_floors = [f for f in floors if f != origin]
    return np.random.choice(possible_floors)

def generate_calls(rate_per_min, origin_dist, sim_time, floors, get_destination_func):
    """
    Generate a list of calls (arrival_time, origin_floor, destination_floor).
    - rate_per_min: calls per minute (float)
    - origin_dist: probability distribution for floors (sum=1)
    - sim_time: total simulation time (minutes)
    """
    calls = []
    t = 0.0
    while t < sim_time:
        # Interarrival ~ Exponential(1 / rate_per_min)
        interarrival = np.random.exponential(1.0 / rate_per_min)
        t += interarrival
        if t > sim_time:
            break
        origin = np.random.choice(floors, p=origin_dist)
        destination = get_destination_func(origin, floors)
        calls.append((t, origin, destination))
    return calls

def simulate_scenario(
    origin_dist, calls_per_hour, sim_time, standby_config,
    floors, elevator_speed, door_time
):
    """
    - origin_dist: list of floats (sum=1), length = #floors
    - calls_per_hour: user-chosen (int)
    - sim_time: total simulation time (minutes)
    - standby_config: tuple of floors where each elevator starts
    - elevator_speed: floors per minute
    - door_time: overhead (minutes) for doors
    """
    num_elevators = len(standby_config)
    # Convert calls/hour -> calls/min
    rate_per_min = calls_per_hour / 60.0

    # 1) Generate calls
    calls = generate_calls(
        rate_per_min=rate_per_min,
        origin_dist=origin_dist,
        sim_time=sim_time,
        floors=floors,
        get_destination_func=default_get_destination
    )

    # 2) Initialize Elevators
    elevators = [Elevator(i, standby_config[i]) for i in range(num_elevators)]

    # 3) Assign calls to best elevator
    waiting_times = []
    for (call_time, origin, destination) in calls:
        best_elevator = None
        best_arrival = float('inf')
        for elevator in elevators:
            start_move_time = max(call_time, elevator.time_next_available)
            arrival_to_origin = start_move_time + travel_time(elevator.current_floor, origin, elevator_speed)
            if arrival_to_origin < best_arrival:
                best_arrival = arrival_to_origin
                best_elevator = elevator
        w = best_elevator.assign_call(call_time, origin, destination, elevator_speed, door_time)
        waiting_times.append(w)

    return np.mean(waiting_times) if waiting_times else 0.0

def run_simulations(
    origin_dist, calls_per_hour, sim_time, floors,
    elevator_speed, door_time, num_elevators
):
    """
    Evaluate all standby configurations for the chosen number of elevators.
    Returns { (floor_0, ..., floor_{n-1}): avg_wait_time }.
    """
    results = {}
    for combo in product(floors, repeat=num_elevators):
        avg_wait = simulate_scenario(
            origin_dist=origin_dist,
            calls_per_hour=calls_per_hour,
            sim_time=sim_time,
            standby_config=combo,
            floors=floors,
            elevator_speed=elevator_speed,
            door_time=door_time
        )
        results[combo] = avg_wait
    return results

# -----------------------------------------
# Streamlit App
# -----------------------------------------

def normalized_distribution(slider_values):
    """Convert the 6 slider values to a distribution that sums to 1."""
    total = sum(slider_values)
    if total <= 0:
        return [1/6]*6
    return [v / total for v in slider_values]

def main():
    st.title("Multi-Elevator Simulation — Using Calls per Hour")

    st.write("""
    This app simulates an N-elevator building (floors 0..5) with 
    **calls per hour** (a more realistic metric). Internally, we convert 
    calls per hour to calls per minute by dividing by 60.
    
    **No same-floor trips**: the destination floor is always different from the origin. 
    All parameter inputs are in collapsible sections in the sidebar.
    """)

    # 1) Basic Setup
    with st.sidebar.expander("Basic Setup", expanded=False):
        num_elevators = st.slider(
            "Number of Elevators",
            min_value=1,
            max_value=4,
            value=2
        )
        sim_time = st.number_input(
            "Simulation Time (minutes)",
            min_value=1,
            max_value=600,
            value=150
        )
        elevator_speed = st.slider(
            "Elevator Speed (floors/minute)",
            min_value=0.5,
            max_value=10.0,
            value=1.0,
            step=0.1
        )
        door_time = st.slider(
            "Door Operation Time (minutes per stop)",
            min_value=0.0,
            max_value=1.0,
            value=0.1,
            step=0.05
        )

    # 2) Floor Distributions
    with st.sidebar.expander("Customize Floor Distributions", expanded=False):
        st.markdown("**Morning Distribution**")
        morning_sliders = []
        default_morning = [0.8, 0.05, 0.05, 0.05, 0.025, 0.025]
        for i in range(6):
            morning_sliders.append(
                st.slider(f"Morning - Floor {i}", 0.0, 1.0, default_morning[i], 0.01)
            )

        st.markdown("**Evening Distribution**")
        evening_sliders = []
        default_evening = [0.05, 0.05, 0.1, 0.25, 0.25, 0.3]
        for i in range(6):
            evening_sliders.append(
                st.slider(f"Evening - Floor {i}", 0.0, 1.0, default_evening[i], 0.01)
            )

        st.markdown("**Non-Peak Distribution**")
        nonpeak_sliders = []
        for i in range(6):
            nonpeak_sliders.append(
                st.slider(f"Non-Peak - Floor {i}", 0.0, 1.0, 1/6, 0.01)
            )

    # Normalize
    MORNING_ORIGIN_DIST = normalized_distribution(morning_sliders)
    EVENING_ORIGIN_DIST = normalized_distribution(evening_sliders)
    NONPEAK_ORIGIN_DIST = normalized_distribution(nonpeak_sliders)

    # 3) Scenario & Calls per Hour
    with st.sidebar.expander("Scenario & Call Volumes (per hour)", expanded=False):
        scenario = st.selectbox(
            "Select a Scenario",
            ["Morning Peak", "Evening Peak", "Non-Peak"]
        )

        st.markdown("**Number of Calls per Hour**")
        morning_calls_hour = st.slider(
            "Morning",
            min_value=0,
            max_value=100,
            value=20,
            step=1
        )
        evening_calls_hour = st.slider(
            "Evening",
            min_value=0,
            max_value=100,
            value=10,
            step=1
        )
        nonpeak_calls_hour = st.slider(
            "Non-Peak",
            min_value=0,
            max_value=100,
            value=5,
            step=1
        )

    st.write(f"**Current Scenario:** {scenario}")

    floors = list(range(6))  # floors [0..5]

    # Button to run simulation
    if st.button("Run Simulation"):
        st.write("Running simulation...")

        # Determine which distribution & call volume to use
        if scenario == "Morning Peak":
            origin_dist = MORNING_ORIGIN_DIST
            calls_hour = morning_calls_hour
        elif scenario == "Evening Peak":
            origin_dist = EVENING_ORIGIN_DIST
            calls_hour = evening_calls_hour
        else:
            origin_dist = NONPEAK_ORIGIN_DIST
            calls_hour = nonpeak_calls_hour

        results = run_simulations(
            origin_dist=origin_dist,
            calls_per_hour=calls_hour,
            sim_time=sim_time,
            floors=floors,
            elevator_speed=elevator_speed,
            door_time=door_time,
            num_elevators=num_elevators
        )

        # Sort results by ascending wait
        combos = list(results.keys())
        combos_str = [str(c) for c in combos]
        avg_waits = list(results.values())
        sorted_idx = np.argsort(avg_waits)
        combos_sorted = [combos_str[i] for i in sorted_idx]
        waits_sorted = [avg_waits[i] for i in sorted_idx]

        # Plot
        fig, ax = plt.subplots(figsize=(10, 6))
        x_vals = range(len(combos_sorted))
        ax.bar(x_vals, waits_sorted, color="skyblue")
        ax.set_xticks(x_vals)
        ax.set_xticklabels(combos_sorted, rotation=90)
        ax.set_xlabel(f"Standby Configurations (Elevator0..Elevator{num_elevators-1})")
        ax.set_ylabel("Average Waiting Time (minutes)")
        ax.set_title(f"Results: {scenario} with {num_elevators} Elevators")
        plt.tight_layout()
        st.pyplot(fig)

        # Best config
        best_i = np.argmin(waits_sorted)
        st.success(
            f"**Best standby config:** {combos_sorted[best_i]} "
            f"(avg wait = {waits_sorted[best_i]:.2f} min)"
        )
    else:
        st.info("Set your parameters and click 'Run Simulation'.")

if __name__ == "__main__":
    main()
