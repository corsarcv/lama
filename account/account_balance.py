
class AccountBalance:

    _instance = None  # Singleton
    _initialized = False
    TEST_START_BALANCE_DEFAULT = 100000.0;
    TEST_TRADE_PRICE = 1000.0

    def __new__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self, verbose=False):
        if not self._initialized:
            self._balance = self.TEST_START_BALANCE_DEFAULT
            self._initialized = True
            self._reset_stats()
        self.verbose = verbose
    
    def reset_balance(self):
        self.print('Reseting account balance. TEST ONLY!')
        self._balance = self.TEST_START_BALANCE_DEFAULT
        self._reset_stats()

    def _reset_stats(self):
        self._min = self._balance
        self._max = self._balance
        self._count = 0
    
    @property
    def trade_amount(self):
        return self.TEST_TRADE_PRICE  # TODO: Change pricing logic
    
    @property
    def balance(self):
        return self._balance
    
    @property
    def in_money(self): # Test only
        return self.balance > self.TEST_START_BALANCE_DEFAULT
    
    @property
    def pnl(self): # Test Only
        return self.balance - self.TEST_START_BALANCE_DEFAULT

    @property
    def stats(self):
        return {
            'current_balance': self._balance, 
            'minimum': self._min,
            'maximum': self._max,
            'transactions_count': self._count}
    
    def open_position(self, amount): 
        if self._balance >= amount:
            self._balance -= amount
            self._count += 1
            if self._balance < self._min:  # TODO: Handle SellShort
                self._min = self._balance
            self.print(f'Opening a position for {amount}. Current balance: {self._balance}')
            return True
        else:
            self.print("Out of balance")
            return False
    
    def close_position(self, amount):
        self._balance += amount
        self._count += 1
        if self._balance > self._max:  # TODO: Handle SellShort
            self._max = self._balance
        self.print(f'Closing a position for {amount}. Current balance: {self._balance}')
        return True

    def print(self, msg):
        if self.verbose:
            print(msg)

    
