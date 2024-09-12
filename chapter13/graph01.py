# Impor pustaka dasar
import json
import pandas as pd
import traceback

# Impor pustaka pydantic
from typing import TypedDict, Annotated, Sequence
import operator

# Impor fungsi kustom
from util_functions import get_last_chains, save_new_chain

# Impor pustaka langchain
from langchain_openai import ChatOpenAI
from langchain_core.messages import BaseMessage, FunctionMessage, HumanMessage
from langchain_core.tools import tool
from langchain_core.utils.function_calling import convert_to_openai_function
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain import hub

# Impor pustaka langgraph
from langgraph.prebuilt import ToolExecutor, ToolInvocation
from langgraph.graph import StateGraph, END

from dotenv import load_dotenv

#### Environment ####
load_dotenv()  # Muat file variabel lingkungan (.env).

# Mempersiapkan input untuk template prompt.
df_list = []
for i in range(6):
    # Membaca file 'data/coworker{i}.csv' dan mengonversinya menjadi DataFrame, lalu membuat salinannya.
    df_list.append(pd.read_csv(f"data/coworker{i}.csv", index_col=0).copy())

# Membuat df_dic yang akan digunakan dalam fungsi eval() di evaluate_pandas_chain.
df_dic = {}
for i, dataframe in enumerate(df_list):
    # Menyimpan setiap DataFrame ke dalam df_dic dengan kunci dalam format 'df1', 'df2', ...
    df_dic[f"df{i + 1}"] = dataframe

# Membuat string yang menjelaskan setiap DataFrame.
questions_str = """
df1: Pilih rekan kerja yang Anda akui sebagai pemain paling berharga (MVP) di perusahaan karena kontribusi mereka yang luar biasa terhadap kinerja dan produktivitas perusahaan.
df2: Pilih rekan kerja yang menurut Anda memiliki potensi terbesar yang belum dimanfaatkan.
df3: Dengan siapa Anda paling suka berkolaborasi?
df4: Dengan rekan kerja mana Anda merasa paling tegang atau mengalami konflik terbanyak?
df5: Siapa rekan kerja yang paling sulit untuk diajak berkolaborasi?
df6: Metadata tentang orang-orang yang berpartisipasi dalam survei
"""

# Parser untuk rantai aksi
def get_action(actions):
    if "<BEGIN>" in actions:  # Jika aksi mengandung "<BEGIN>"
        # Membagi berdasarkan tanda "->" dan mengambil elemen kedua, lalu menghapus spasi.
        a = actions.split("->")[1].strip()
    else:  # Jika aksi tidak mengandung "<BEGIN>"
        # Membagi berdasarkan tanda "->" dan mengambil elemen pertama, lalu menghapus spasi.
        a = actions.split("->")[0].strip()
    return a  # Mengembalikan aksi yang diekstraksi.

# Fungsi untuk mengevaluasi aksi berikutnya dalam rantai
@tool
def evaluate_pandas_chain(
    chain: Annotated[
        str,
        "Rantai aksi pandas. Contoh: df1.groupby('age').mean() -> df1.sort_values() -> <END>",
    ],
    inter=None,
):
    """Gunakan fungsi ini untuk menjalankan rantai kode pandas pada DataFrame"""

    name = "evaluate_pandas_chain"

    try:
        action = get_action(chain)
        print("\n\naction: ", action)

        inter = eval(action, {"inter": inter, "df_dic": df_dic})

        if isinstance(inter, pd.DataFrame):
            intermediate = inter.head(50).to_markdown()
        else:
            intermediate = inter

        return intermediate, action, inter

    except Exception as e:
        return f"Terjadi pengecualian: {traceback.format_exc()}", action, None


# Fungsi untuk melihat DataFrame


@tool
def view_pandas_dataframes(
    df_list: Annotated[
        Sequence[str],
        "Daftar hingga 3 DataFrame pandas yang ingin dilihat. Contoh: [df1, df2, df3]",
    ]
):
    """Gunakan fungsi ini untuk melihat head(10) dari DataFrame untuk menjawab pertanyaan"""

    name = "view_pandas_dataframes"

    markdown_str = "Berikut adalah .head(10) dari DataFrame yang diminta:\n"
    for df in df_list:
        df_head = df_dic[df].head(10).to_markdown()
        markdown_str += f"{df}:\n{df_head}\n"

    markdown_str = markdown_str.strip()
    return markdown_str


tools = [evaluate_pandas_chain, view_pandas_dataframes]
tool_executor = ToolExecutor(tools)

functions = [convert_to_openai_function(t) for t in tools]

# Mengambil prompt dari langchain hub.
SYSTEM_PROMPT = hub.pull("hrubyonrails/multi-cot").messages[0].prompt.template

# Mencetak template prompt yang telah diformat.
print(SYSTEM_PROMPT)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            SYSTEM_PROMPT,  # Mengatur sistem prompt.
        ),
        MessagesPlaceholder(
            variable_name="messages"
        ),  # Mengatur placeholder pesan.
    ]
)

# Menetapkan panjang daftar DataFrame ke variabel num_dfs.
prompt = prompt.partial(num_dfs=len(df_list))
# Menetapkan nama alat yang dipisahkan dengan koma ke variabel tool_names.
prompt = prompt.partial(tool_names=", ".join([tool.name for tool in tools]))
# Menetapkan string pertanyaan ke variabel questions_str.
prompt = prompt.partial(questions_str=questions_str)

# Mengirimkan kueri yang berhasil sebelumnya.
chain_examples = ""
# Memeriksa apakah chain terakhir dalam format DataFrame.
if type(get_last_chains()) == pd.core.frame.DataFrame:
    # Mengulangi kueri dan kolom chain.
    for index, row in get_last_chains().iterrows():
        # Membuat contoh chain.
        chain_examples += f'Question: {row["query"]}\nChain: {row["chain"]}\n\n'
# Menetapkan contoh chain ke variabel chain_examples.
prompt = prompt.partial(chain_examples=chain_examples)

# Binding model
model = prompt | ChatOpenAI(
    model="gpt-4-0125-preview").bind_functions(functions)

# Membuat status grafis


class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], operator.add]  # Urutan pesan
    actions: Annotated[Sequence[str], operator.add]  # Urutan aksi
    inter: pd.DataFrame  # DataFrame sementara
    question: str  # Pertanyaan
    memory: str  # Memori

# Mendefinisikan fungsi untuk memutuskan apakah harus melanjutkan: edge kondisional
def should_continue(state):
    messages = state['messages']
    last_message = messages[-1]
    # Jika tidak ada pemanggilan fungsi, hentikan
    if "function_call" not in last_message.additional_kwargs:
        return "end"
    # Jika ada pemanggilan fungsi, lanjutkan ke node call_tool
    else:
        return "continue"

# Mendefinisikan fungsi untuk memanggil model
def call_model(state):
    response = model.invoke(state)
    # Mengembalikan dalam bentuk daftar, yang akan ditambahkan ke daftar yang ada
    return {"messages": [response]}

# Mendefinisikan fungsi untuk menjalankan alat
def call_tool(state):
    messages = state['messages']

    # Berdasarkan kondisi continue
    # Dapat dilihat bahwa pesan terakhir mencakup pemanggilan fungsi
    last_message = messages[-1]

    tool_input = last_message.additional_kwargs["function_call"]["arguments"]

    tool_input_dict = json.loads(tool_input)
    tool_input_dict['inter'] = state['inter']

    if last_message.additional_kwargs["function_call"]["name"] == 'view_pandas_dataframes':
        # Membuat ToolInvocation dari function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=tool_input_dict,
        )
        # Memanggil tool_executor dan mendapatkan respons
        response = tool_executor.invoke(action)

        function_message = FunctionMessage(content=str(response), name=action.tool)
        return {"messages": [function_message]} # ,"actions": [attempted_action]}

    # Jika alat mengevaluasi rantai
    elif last_message.additional_kwargs["function_call"]["name"] == 'evaluate_pandas_chain':
        # Membuat ToolInvocation dari function_call
        action = ToolInvocation(
            tool=last_message.additional_kwargs["function_call"]["name"],
            tool_input=tool_input_dict,
        )
        # Memanggil tool_executor dan mendapatkan respons
        response, attempted_action, inter = tool_executor.invoke(action)

        if "An exception occured:" in str(response):
            error_info = f"""
            Tindakan yang telah dilakukan sebelumnya: 
            {state['actions']}

            Tindakan saat ini: 
            {attempted_action}

            Hasil .head(50): 
            {response}

            Anda harus memperbaiki pendekatan dan melanjutkan hingga Anda dapat menjawab pertanyaan berikutnya:
            {state['question']}

            Teruskan rantai dengan format berikut: action_i -> action_i+1 ... -> <END>
            """
            print(error_info)

            function_message = FunctionMessage(content=str(error_info), name=action.tool)
            return {"messages": [function_message]}

        else:
            success_info = f"""
            Tindakan yang telah dilakukan sebelumnya: 
            {state['actions']}

            Tindakan saat ini: 
            {attempted_action}

            Hasil .head(50):
            {response}

            Anda harus melanjutkan hingga Anda dapat menjawab pertanyaan berikutnya:
            {state['question']}

            Teruskan rantai dengan format berikut: action_i -> action_i
            """
# Mendefinisikan graf baru.
workflow = StateGraph(AgentState)

# Mendefinisikan dua node yang akan berputar.
workflow.add_node("agent", call_model)
workflow.add_node("action", call_tool)

# Menetapkan titik masuk ke `agent`.
# Ini berarti node ini akan dipanggil pertama kali.
workflow.set_entry_point("agent")

# Sekarang tambahkan tepi bersyarat.
workflow.add_conditional_edges(
    # Pertama, mendefinisikan node awal. Gunakan `agent`.
    # Ini berarti tepi ini dilakukan setelah node `agent` dipanggil.
    "agent",
    # Kemudian, sediakan fungsi untuk menentukan node yang akan dipanggil berikutnya.
    should_continue,
    # Terakhir, berikan pemetaan.
    # Kunci adalah string dan nilai adalah node lain.
    # END adalah node khusus yang menunjukkan graf harus diakhiri.
    # Output dari `should_continue` kemudian dibandingkan dengan kunci pemetaan ini
    # untuk memanggil node yang cocok.
    {
        # Jika `continue`, panggil node `action`.
        "continue": "action",
        # Jika tidak, akhiri.
        "end": END,
    },
)

# Sekarang tambahkan tepi umum dari `tools` ke `agent`.
# Ini berarti setelah `tools` dipanggil, node `agent` akan dipanggil berikutnya.
workflow.add_edge("action", "agent")

# Terakhir, kompilasi!
# Ini dikompilasi menjadi LangChain Runnable,
# sehingga dapat digunakan dengan cara yang sama seperti hal lainnya.
app = workflow.compile()

# Mari coba beberapa pertanyaan yang melibatkan beberapa langkah dan tabel yang berbeda.

user_query = "Tampilkan berapa kali Steven Rollins terpilih sebagai MVP. Juga, tunjukkan jumlah peringkat yang diterima oleh karyawan ini untuk setiap alasan MVP."
# user_query = "Tim mana yang menerima suara terbanyak sebagai tim yang sulit diajak bekerja sama?"

# Mengatur data input dalam bentuk dictionary. Termasuk kueri pengguna, aksi, memori, dll.
inputs = {"messages": [HumanMessage(content=user_query)], "actions": [
    "<BEGIN>"], "question": user_query, "memory": ""}

# Menggunakan fungsi app.stream() untuk melakukan streaming data input, dan atur batas rekursi menjadi 40.
for output in app.stream(inputs, {"recursion_limit": 40}):
    for key, value in output.items():  # Mengulangi setiap pasangan kunci-nilai dalam dictionary output.
        if key == "agent":  # Jika kunci adalah "agent"
            print("🤖 Agen sedang bekerja...")  # Menampilkan pesan "Agen sedang bekerja..."
        elif key == "action":  # Jika kunci adalah "action"
            # Jika nama pesan aksi adalah "view_pandas_dataframes"
            if value["messages"][0].name == "view_pandas_dataframes":
                print("🛠️ Aksi saat ini:")  # Menampilkan pesan "Aksi saat ini:"
                # Menampilkan pesan "viewing dataframes"
                print("`viewing dataframes`")
            else:  # Jika tidak
                if "actions" in value.keys():  # Jika kunci "actions" ada
                    # Menampilkan pesan "Aksi saat ini:"
                    print(f"🛠️ Aksi saat ini:")
                    print(f"`{value['actions']}`")  # Menampilkan aksi saat ini
                    print(f"Output saat ini:")  # Menampilkan pesan "Output saat ini:"
                    print(value["inter"])  # Menampilkan output saat ini
                else:  # Jika kunci "actions" tidak ada
                    # Menampilkan pesan kesalahan dan mencoba lagi
                    print(f"⚠️ Terjadi kesalahan, mencoba lagi...")
        else:  # Jika kunci lainnya
            print("🏁 Menyelesaikan...")  # Menampilkan pesan "Menyelesaikan..."
            print(f"Output akhir:")  # Menampilkan pesan "Output akhir:"
            print(value["inter"])  # Menampilkan output akhir
            print(f"Rantai aksi akhir:")  # Menampilkan pesan "Rantai aksi akhir:"
            # Menampilkan rantai aksi akhir
            print(" -> ".join(value["actions"]) + ' -> <END>')

        print("---")  # Menampilkan garis pemisah
        pass  # Tidak melakukan apa-apa

output_dict = output["__end__"]  # Mengambil nilai untuk kunci "__end__" dari dictionary output
agent_response = output_dict["messages"][-1].content  # Mengambil respons agen
final_table = output_dict["inter"]  # Mengambil tabel akhir
# Menghapus tag '<END>' dari respons agen untuk menghasilkan pesan akhir
final_message = agent_response.replace('<END>', '')

# Mencetak pesan terakhir.
print(final_message)

print(final_table)  # Keluarkan tabel akhir.