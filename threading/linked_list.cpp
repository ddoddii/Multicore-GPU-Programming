template <typename T> class linked_list_queue
{
    class node
    {
      public:
        node(T init_value) : value(init_value)
        {
        }
        T value;
        node *next;
    };

    node *head;
    node *tail;

  public:
    linked_list_queue()
    {
        head = tail = new node(0); // dummy node
    }

    void push(T value)
    {
        node *tmp = new Node(value);
        tmp->value = value;
        tmp->next = nullptr;
        tail->next = tmp;
        tail = tmp;
    }

    bool pop(T &value)
    {
        node *old_head = head;
        node *new_head = old_head->next;
        if (new_head == nullptr)
        {
            return false;
        }
        value = new_head->value;
        head = new_head;
        delete old_head;

        return true;
    }
};