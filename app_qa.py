import streamlit as st
from question_generator import generate_questions, daily_tasks
from models import question_input

st.set_page_config(page_title="User Input Form", layout="centered")
st.title("User Information Form")

# Create a form for user input
with st.form("user_info_form"):
    # Text input for name
    name = st.text_input("Name", placeholder="Enter your name")
    
    # Number input for age
    age = st.number_input("Age", min_value=0, max_value=150, step=1)
    
    # Selectbox for gender
    gender = st.selectbox("Gender", ["Male", "Female", "Other", "Prefer not to say"])
    
    # Selectbox for topics
    topic = st.selectbox("Select a Topic", ["Math", "Indian Mythology", "India News", "Cricket"])
    
    # Submit button
    submitted = st.form_submit_button("Submit")

# Display the input after form submission
if submitted:
    st.subheader("Your Information:")
    st.write(f"**Name:** {name}")
    st.write(f"**Age:** {age}")
    st.write(f"**Gender:** {gender}")
    st.write(f"**Topic:** {topic}")
    tasks_input = question_input(topic=topic, age=age, gender=gender)
   
    
    # Call the question_generator tool with the topic
   # question_gen_input = question_input(topic=topic)
    result = generate_questions.invoke({
    "input_data": {
        "topic": topic
    }
})
    
    # store result in session state so it persists across reruns
    st.session_state['qa_result'] = result

# if we already generated a question, show it along with the options and let the user answer
if 'qa_result' in st.session_state:
    qa = st.session_state['qa_result']
    st.subheader("Generated Question:")
    st.write(f"**Question:** {qa.question}")

    # display options as radio buttons
    user_choice = st.radio("Select the correct answer", qa.options, key="user_choice")

    if st.button("Submit Answer"):
        if user_choice == qa.answer:
            st.success("✅ Correct! Well done.")
        else:
            st.error(f"❌ Wrong. The correct answer is: {qa.answer}")

st.write("Generating daily tasks based on your input...")
tasks_result = daily_tasks.invoke({
    "input_data": tasks_input.model_dump()
})
st.write("Here are some daily tasks suitable for you:")
for idx, task in enumerate(tasks_result.tasks, 1):
    st.write(f"{idx}. {task}")  