import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss, MSELoss, BCELoss
import networkx as nx
import time


torch.manual_seed(43)

torch.autograd.set_detect_anomaly(True)
import re



class sample(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        # x = torch.bernoulli(x)
        # x = (torch.sign(x) + 1.)/2.
        return x

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        #print(grad_output)
        return grad_output * 10


sampler = sample.apply

def AND(*args):
    return torch.prod(torch.stack(args, dim = 1).squeeze(1), dim = 1)

def NAND(*args):
    return 1 - torch.prod(torch.stack(args, dim = 1).squeeze(1), dim = 1)

def OR(*args):
    return 1 - torch.prod(1 - torch.stack(args, dim = 1).squeeze(1), dim = 1)

def NOR(*args):
    return torch.prod(1 - torch.stack(args, dim = 1).squeeze(1), dim = 1)

def XOR(a, b):
    return 1 - (1 - a * (1 - b)) * (1 - (1 - a) * b) 

def XNOR(a, b):
    return 1 - (1 - (1 - a) * (1 - b)) * (1 - a * b) 

def NOT(a):
    return 1 - a

def read_verilog_file(file_path):
    with open(file_path, 'r') as file:
        verilog_string = file.read()
    return verilog_string

def extract_variables(expr, inputs, registers):
    # Regular expression pattern to match variables inside parentheses
    pattern = r'\((.*?)\)'
    # Find all matches of the pattern in the expression
    matches = re.findall(pattern, expr)
    # Split each match by commas and strip to get individual variables
    variables = [var.strip() for match in matches for var in match.split(',')]
    #variables = [var for var in variables if not var.startswith('self.')]
    ##variables = [var for var in variables if var not in registers]
    ##variables = [var for var in variables if var not in inputs]
    return variables

def refine(text, registers, inputs):
    lines = text.strip().split('\n')
    filtered_lines = [line for line in lines if not line.startswith('#') and 'INPUT' not in line and 'OUTPUT' not in line and 'DFF' not in line]
    registered_lines = [line for line in lines if 'DFF' in line]
    
    modified_lines = []
    states = []
    for line in registered_lines:
        variable, value = line.split('=')        
        modified_variable = f"{variable.strip()}"        
        value = value.split('(')[1].split(')')[0]
        states.append(value)
        modified_line = f"{modified_variable} = {value}"
        modified_lines.append(modified_line)
    
    #filtered_lines = [line for line in filtered_lines if 'INPUT' not in line and 'OUTPUT' not in line]
    filtered_string = '\n'.join(filtered_lines)
    non_empty_lines = [line for line in filtered_string.split('\n') if line.strip() != '']
    filtered_string = '\n'.join(non_empty_lines)

    '''for variable in registers:
        if variable in filtered_string:
            #filtered_string = filtered_string.replace(variable, f'self.{variable}')
            filtered_string = re.sub(rf'\b{variable}\b', f'self.{variable}', filtered_string)'''
    
    # for line in filtered_string.strip().split('\n'):
    #     variables = extract_variables(line)

    dependencies = {}
    expressions = {}
    # Extract variable names and their dependencies
    for line in filtered_string.strip().split('\n'):
        parts = line.split('=')
        variable = parts[0].strip()
        expr = parts[1].strip() if len(parts) > 1 else ""
        dependencies[variable] = extract_variables(expr, inputs, registers)
        expressions[variable] = parts[1].strip()
    G = nx.DiGraph()
    for variable, deps in dependencies.items():
        for dep in deps:
            G.add_edge(dep, variable)
    # Topological sorting of statements
    sorted_vars = list(nx.topological_sort(G))
    rearranged_statements = [f"{var} = {expressions[var]}" for var in sorted_vars if var not in inputs and var not in registers]
    modified_input_string = '\n'.join(rearranged_statements)
    print(len(dependencies), len(rearranged_statements))
    
    return modified_input_string.strip().split('\n'), states #modified_lines



def parse_verilog_module(verilog_code):
    lines = verilog_code.split('\n')
    inputs = []
    outputs = []
    registers = []
    
    for line in lines:
        
        if 'INPUT' in line:
            if line.startswith('INPUT('):
                # Extract the variable name between parentheses
                variable = line.split('(')[1].split(')')[0].replace('.', '')
                inputs.append(variable)
        elif 'OUTPUT' in line:
            if line.startswith('OUTPUT('):
                # Extract the variable name between parentheses
                variable = line.split('(')[1].split(')')[0].replace('.', '')
                outputs.append(variable)
        elif 'DFF' in line:
            variable = line.split('=')[0].strip().replace('.', '')
            registers.append(variable)
    return inputs, outputs, registers

def generate_pytorch_model(module_name, inputs, outputs, registers, assignments):
    class_name = module_name.lower()
    class_definition = f"class {class_name}(nn.Module):\n" \
                       f"    def __init__(self, batch_size, device):\n" \
                       f"        super().__init__()\n"
    class_definition += f"        self.batch_size = batch_size\n"
    class_definition += f"        self.device = device\n"
    '''for register in registers:
        class_definition += f"        self.{register} = torch.zeros((batch_size, 1), device = device)\n"
    class_definition += "\n"
    class_definition += f"    def reset_registers(self):\n"
    for register in registers:
        class_definition += f"        self.{register} = self.{register} * 0\n"
    class_definition += "\n"
    class_definition += f"    def detach_registers(self):\n"
    for register in registers:
        class_definition += f"        self.{register} = self.{register}.detach()\n"'''
    class_definition += f"    def init_registers(self):\n"
    for register in registers:
        class_definition += f"        {register} = torch.zeros((self.batch_size, 1), device = self.device)\n"
    class_definition += f"        return {', '.join(registers)}\n"
    class_definition += "\n"
    class_definition += f"    def forward(self, inputs, registers):\n"
    class_definition += f"        {', '.join(inputs)} = inputs\n"
    class_definition += f"        {', '.join(registers)} = registers\n"
    logics_assignments, registers_assignments = refine(assignments, registers, inputs)
    for assignment in logics_assignments:
        class_definition += f"        {assignment}\n"
    '''for assignment in registers_assignments:
        class_definition += f"        {assignment}\n"'''
    registers_assignments = [f"sampler({assignment})" for assignment in registers_assignments]
    class_definition += f"        states = {', '.join(registers_assignments)}\n"
    class_definition += f"        outputs = {', '.join(outputs)}\n"
    class_definition += "\n" \
                       f"        return outputs, states\n"
    return class_definition



class PIEmbedding(nn.Module):
    def __init__(
        self,
        input_names: list[str],
        input_shape: list[int],
        device: str = "cpu",
        batch_size: int = 1,
    ):
        super().__init__()
        self.input_shape = input_shape
        self.device = device
        self.batch_size = batch_size
        self.input_names = input_names

        self.parameters_list = nn.ParameterList()
        for name, size in zip(input_names, input_shape):
            param = nn.Parameter(torch.randn(batch_size, size, device=device))
            self.parameters_list.append(param)

        '''for name, size in zip(input_names, input_shape):
            code_string = f"self.{name} = torch.nn.parameter.Parameter(torch.randn(batch_size, {size}, device=device))"
            exec(code_string)'''

                
        self.activation = torch.nn.Sigmoid()  

    def forward(self, idx):
        outputs = []
        for param in self.parameters_list:
            param[:,idx].data.clamp_(-3.5, 3.5)
            output_tensor = self.activation(2 * param[:,idx].unsqueeze(-1))
            outputs.append(output_tensor)
        return outputs
        
        '''for name in self.input_names:
            code_string1 = f"self.{name}.data.clamp_(-3.5, 3.5)"
            exec(code_string1)
            code_string2 = f"{name} = self.activation( 2 * self.{name} )"
            exec(code_string2)
        input_names_string = ", ".join(self.input_names)
        code_string = f"return {input_names_string}"
        exec(code_string)'''


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    batch_size = 7500
    num_train_epochs = 2
    lr = 15
    NO_clk_cycles = 25
    file_path = './ISCAS89/s38584.bench'
    bench_code = read_verilog_file(file_path).replace('.', '_')
    inputs, outputs, registers = parse_verilog_module(bench_code)
    num_outputs = len(outputs)
    # refined_verilog_code = refine(verilog_code)
    # print(refined_verilog_code)
    # assignments = re.findall(r'assign\s+(\S+)\s*=\s*(\S+)\s*([&|^])\s*(\S+)\s*;', verilog_code.replace("\\", "").replace("[", "[:,"))
    module_name = file_path.split('/')[-1].replace('.bench', '').replace('.', '')
    pytorch_model = generate_pytorch_model(module_name, inputs, outputs, registers, bench_code)
    with open('sin.txt', 'w') as file:
        # Write a string to the file
        file.write(pytorch_model)
    exec(pytorch_model)
    #Circuit =  exec(f"{module_name}()")
    class_object = locals()[module_name]

    Circuit = class_object(batch_size, device)
    Circuit_inputs = PIEmbedding(inputs, [NO_clk_cycles] * len(inputs), device, batch_size)
    '''input_vars = ", ".join(inputs)
    code_string = f"{input_vars} = Circuit_inputs()"
    exec(code_string)'''
    loss = MSELoss(reduction = 'sum')
    optim = torch.optim.SGD([
                {'params': Circuit.parameters()},
                {'params': Circuit_inputs.parameters()}
            ], lr=lr)
    target_list = []

    for name, size in zip(outputs, [1] * len(outputs)):
            target_list.append(torch.randint(2, (batch_size, size), device=device).float())
    
    '''with torch.no_grad():
        inputs_list = []
        for name, size in zip(inputs, [1] * len(inputs)):
                inputs_list.append(torch.randint(2, (batch_size, size), device=device).float())
        states = Circuit.init_registers()
        for idx in range(NO_clk_cycles):

            target_list, states = Circuit(inputs_list, states)'''

    if module_name in ['s27', 's2081', 's298','s382', 's386', 's400', 's4201', 's444', 's510', 's526', 's526n', 's635', 's8381', 's938', 's1423']:
        target = torch.ones((batch_size, 1), device=device) #torch.cat((d[0],d[2]), dim = -1)
    elif module_name in ['s344', 's349', 's1196', 's1238', 's1269', 's3271', 's4863']:
        target = torch.cat((torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device) ), dim = -1)
    elif module_name in ['s499', 's641', 's713', 's820', 's832', 's953', 's967', 's991', 's1488', 's1494', 's1512', 's3384', 's5378', 's6669', 's92341', 's9234']:
        target = torch.cat((torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device) ), dim = -1)
    else:
        target = torch.cat((torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device), torch.zeros((batch_size, 1), device=device), torch.ones((batch_size, 1), device=device)), dim = -1)
    

    start = time.perf_counter()
    for epoch in range(num_train_epochs):
        

        Circuit.train()
        Circuit_inputs.train()
        optim.zero_grad()
        states = Circuit.init_registers()
        #print(Circuit_inputs.parameters_list[0])
        for idx in range(NO_clk_cycles):
            '''print('*************************************************\n')
            print(states)
            print('*************************************************\n')'''
            outputs = Circuit_inputs(idx)
            '''print('*************************************************\n')
            print(outputs)
            print('*************************************************\n')'''
            outputs_list, states = Circuit(outputs, states)
            states = sampler(states)
        
        total_loss = 0.0
        #print(torch.stack(outputs_list,dim=1).view(-1))
        #print(torch.stack(target_list,dim=1).view(-1))
        if module_name in ['s27', 's2081', 's298','s382', 's386', 's400', 's4201', 's444', 's510', 's526', 's526n', 's635', 's8381', 's938', 's1423']:
            if num_outputs > 1:
                output = outputs_list[-1] #torch.cat((d[0],d[2]), dim = -1)
            else:
                output = outputs_list
        elif module_name in ['s344', 's349', 's1196', 's1238', 's1269', 's3271', 's4863']:
            output = torch.cat(( outputs_list[0], outputs_list[-1] ), dim = -1)
        elif module_name in ['s499', 's641', 's713', 's820', 's832', 's953', 's967', 's991', 's1488', 's1494', 's1512', 's3384', 's5378', 's6669', 's92341', 's9234']:
            output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[-1] ), dim = -1)
        else:
            output = torch.cat(( outputs_list[0], outputs_list[15], outputs_list[31], outputs_list[63], outputs_list[-1] ), dim = -1)

        total_loss = loss(output, target)
        # for output, target in zip(outputs_list, target_list):
            
        #     l = loss(output, target)
        #     #l.requires_grad = True
        #     total_loss += l
        print(total_loss)
        
        total_loss.backward()
        ##Circuit.detach_registers()
        
        optim.step()
    end = time.perf_counter() - start
    print('{:.6f}s for the calculation'.format(end))

    sol_list = []
    for param in Circuit_inputs.parameters_list:
        sol_list.append(param)
    print(torch.cat(sol_list,dim=-1).shape)
    solutions = torch.unique( (torch.sign(torch.cat(sol_list,dim=-1))+1.)/2., dim = 0).flip(dims=[0])
    #solutions = torch.unique(torch.cat(sol_list, dim=-1), dim = 0).flip(dims=[0])
    # print(sol_list)
    
    new_input_list = []

    for i in range(len(inputs)):
        new_input_list.append(solutions[:,i * NO_clk_cycles : (i+1) * NO_clk_cycles])
    # print(new_input_list)

    with torch.no_grad():
        states_pre = Circuit.init_registers()
        states = [tensor[0:solutions.shape[0], :] for tensor in states_pre]
        for idx in range(NO_clk_cycles):
            new_input_list1 = []
            for param in new_input_list:
                new_input_list1.append(param[:,idx].unsqueeze(-1))
            solutions_output, states = Circuit(new_input_list1, states)
    if module_name in ['s27', 's2081', 's298','s382', 's386', 's400', 's4201', 's444', 's510', 's526', 's526n', 's635', 's8381', 's938', 's1423']:
        final_output = solutions_output[-1]
        if num_outputs > 1:
            final_output = solutions_output[-1]#torch.cat((d[0],d[2]), dim = -1)
        else:
            final_output = solutions_output #torch.cat((d[0],d[2]), dim = -1)
    elif module_name in ['s344', 's349', 's1196', 's1238', 's1269', 's3271', 's4863']:
        final_output = torch.cat(( solutions_output[0], solutions_output[-1] ), dim = -1)
    elif module_name in ['s499', 's641', 's713', 's820', 's832', 's953', 's967', 's991', 's1488', 's1494', 's1512', 's3384', 's5378', 's6669', 's92341', 's9234']:
        final_output = torch.cat(( solutions_output[0], solutions_output[15], solutions_output[-1] ), dim = -1)
    else:
        final_output = torch.cat(( solutions_output[0], solutions_output[15], solutions_output[31], solutions_output[63], solutions_output[-1] ), dim = -1)
    print('Number of unique solutions:', torch.eq(final_output, target[0:final_output.shape[0],:]).prod(-1).sum(-1))
    # print(final_output)
    # total_loss = loss(final_output, target)
    # print(total_loss)

if __name__ == "__main__":
    main()