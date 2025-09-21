from extract_parameter import extract_parameter
q_list = ["Let x be a positive real number and f a function of x.",
          "Assume n is an integer and y is a real-valued function of t.",
          "Consider z, a complex variable and u, a function of x and y.",
          ]

for q in q_list:
    print(f"Question: {q}")
    result = extract_parameter(q)
    print(f"Extracted Params: {result}")